import time

import numpy as np

from pandda_gemmi.interfaces import *

from pandda_gemmi.args import PanDDAArgs

from pandda_gemmi.fs import PanDDAFS
from pandda_gemmi.dataset import XRayDataset, StructureArray
from pandda_gemmi.dmaps import (
    SparseDMap,
    SparseDMapStream,
    TruncateReflections,
    SmoothReflections,
)
from pandda_gemmi.alignment import Alignment, DFrame
from pandda_gemmi.processor import ProcessLocalRay, Partial

from pandda_gemmi.comparators import (
    get_comparators,
    FilterRFree,
    FilterSpaceGroup,
    FilterResolution,
    FilterCompatibleStructures
)
from pandda_gemmi.dmaps import save_dmap
from pandda_gemmi.event_model.evaluate import evaluate_model
from pandda_gemmi.event_model.characterization import get_characterization_sets, CharacterizationGaussianMixture, CharacterizationNN
from pandda_gemmi.event_model.outlier import PointwiseNormal
from pandda_gemmi.event_model.cluster import ClusterDensityDBSCAN
from pandda_gemmi.event_model.score import ScoreCNN, get_model_map, ScoreCNNLigand
from pandda_gemmi.event_model.filter import FilterSize, FilterCluster, FilterScore, FilterLocallyHighestScoring, FilterLocallyHighestLargest
from pandda_gemmi.event_model.select import select_model
from pandda_gemmi.event_model.output import output_models, output_events, output_maps

from pandda_gemmi.site_model import HeirarchicalSiteModel, ClusterSites, Site, get_sites

from pandda_gemmi.autobuild import autobuild, AutobuildResult
from pandda_gemmi.autobuild.inbuilt import AutobuildInbuilt
from pandda_gemmi.autobuild.merge import merge_autobuilds, MergeHighestRSCC, MergeHighestBuildScore, MergeHighestBuildAndEventScore
from pandda_gemmi.autobuild.preprocess_structure import AutobuildPreprocessStructure
from pandda_gemmi.autobuild.preprocess_dmap import AutobuildPreprocessDMap

from pandda_gemmi.ranking import rank_events, RankHighEventScore, RankHighBuildScore

from pandda_gemmi.tables import output_tables
from pandda_gemmi.pandda_logging import PanDDAConsole

def process_model(
        ligand_files,
        model_number,
        dataset_dmap_array,
        characterization_set_dmaps_array,
        reference_frame,
        model_map,
        score,
):
    # Get the statical maps
    mean, std, z = PointwiseNormal()(
        dataset_dmap_array,
        characterization_set_dmaps_array
    )

    mean_grid = reference_frame.unmask(SparseDMap(mean))
    z_grid = reference_frame.unmask(SparseDMap(z))
    xmap_grid = reference_frame.unmask(SparseDMap(dataset_dmap_array))
    median = np.median(mean[reference_frame.mask.indicies_sparse_inner_atomic])


    model_grid = reference_frame.unmask(SparseDMap(model_map))

    inner_mask_zmap = z[reference_frame.mask.indicies_sparse_inner_atomic]
    percentage_z_2 = float(np.sum(np.abs(inner_mask_zmap) > 2)) / inner_mask_zmap.size
    print(f"Model number: {model_number}: z > 2: {percentage_z_2}")

    # Initial
    events = ClusterDensityDBSCAN()(z, reference_frame)
    print(f"Initial events: {len(events)}")
    j = 0
    for event in sorted(events.values(), key=lambda _event: _event.pos_array.size, reverse=True):
        event_centroid_array = np.mean(event.pos_array, axis=0).flatten()
        print(f"\t\t\tEvent {j}: size: {event.pos_array.size}: centroid: {event_centroid_array[0]} {event_centroid_array[1]} {event_centroid_array[2]}")
        j += 1
        if j > 5:
            break

    if len(events) == 0:
        return None, None, None

    # Filter the events pre-scoring
    for filter in [
        FilterSize(reference_frame, min_size=5.0),
        # FilterCluster(5.0),
    ]:
        events = filter(events)

    print(f"After filer size and cluster: {len(events)}")

    if len(events) == 0:
        return None, None, None

    # Score the events
    time_begin_score_events = time.time()
    events = score(ligand_files, events, xmap_grid, mean_grid, z_grid, model_grid,
                   median,
                   )
    time_finish_score_events = time.time()
    print(f"\t\t\tScored events in: {round(time_finish_score_events - time_begin_score_events, 2)}")

    # Filter the events post-scoring
    for filter in [
        FilterScore(0.30),
        # FilterLocallyHighestScoring(8.0),
        FilterLocallyHighestLargest(5.0),

    ]:
        events = filter(events)
    print(f"After filter score: {len(events)}")

    if len(events) == 0:
        return None, None, None

    return events, mean, z


def pandda(args: PanDDAArgs):
    time_pandda_begin = time.time()
    console = PanDDAConsole()
    console.start_pandda()
    console.start_parse_command_line_args()
    console.summarise_arguments(args)

    # Get the processor
    console.start_initialise_multiprocessor()
    processor: ProcessorInterface = ProcessLocalRay(args.local_cpus)
    console.print_initialized_local_processor(args)

    # Get the FS
    console.start_fs_model()
    fs: PanDDAFSInterface = PanDDAFS(Path(args.data_dirs), Path(args.out_dir))
    console.summarise_fs_model(fs)

    # Get the scoring method
    score = ScoreCNNLigand()

    # Get the datasets
    console.start_load_datasets()
    datasets: Dict[str, DatasetInterface] = {
        dataset_dir.dtag: XRayDataset.from_paths(
            dataset_dir.input_pdb_file,
            dataset_dir.input_mtz_file,
            dataset_dir.input_ligands,
        )
        for dataset_dir
        in fs.input.dataset_dirs.values()
    }
    dataset_refs = {_dtag: processor.put(datasets[_dtag]) for _dtag in datasets}
    structure_array_refs = {_dtag: processor.put(StructureArray.from_structure(datasets[_dtag].structure)) for _dtag in
                            datasets}
    console.summarise_datasets(datasets, fs)

    # Process each dataset
    pandda_events = {}
    time_begin_process_datasets = time.time()
    console.start_process_shells()
    for dtag in datasets:

        # if dtag != "JMJD2DA-x427":
        #     continue

        # if dtag != "JMJD2DA-x427":
        #     continue

        # if dtag not in [
        #     "JMJD2DA-x427",
        #     "JMJD2DA-x379",
        #     "JMJD2DA-x585",
        #     "JMJD2DA-x533",
        #     "JMJD2DA-x387"
        # ]:
        #     continue
        # if dtag not in [
        #     "SETDB1-x223",
        #     "SETDB1-x218"
        # ]:
        #     continue

        # if dtag not in [
        #     "JMJD2DA-x427",
        #     "JMJD2DA-x353"
        # ]:
        #     continue

        # if dtag not in [
        #     "Mpro-i0206"
        # ]:
        #     continue

        time_begin_process_dataset = time.time()

        # Get the dataset
        dataset = datasets[dtag]

        # Get the resolution of the dataset
        dataset_res = dataset.reflections.resolution()
        print(f"Dataset resolution is: {dataset.reflections.resolution()}")

        # Get the comparator datasets
        comparator_datasets: Dict[str, DatasetInterface] = get_comparators(
            datasets,
            [
                FilterRFree(0.4),
                FilterSpaceGroup(dataset),
                FilterCompatibleStructures(dataset),
                FilterResolution(dataset_res, 60, 0.1)]
        )
        if len(comparator_datasets) < 30:
            print(f"NOT ENOUGH COMPARATOR DATASETS! SKIPPING!")
            continue


        # Get the resolution to process at
        processing_res = max(
                    [_dataset.reflections.resolution() for _dataset in comparator_datasets.values()]
            )
        print(f"Processing res is: {processing_res}")
        print(f"Number of comparator datasets: {len(comparator_datasets)}")

        # Ensure the dataset itself is included in comparators
        if dtag not in comparator_datasets:
            comparator_datasets[dtag] = dataset

        # Get the alignments, and save them to the object store
        time_begin_get_alignments = time.time()
        alignments: Dict[str, AlignmentInterface] = processor.process_dict(
            {_dtag: Partial(Alignment.from_structure_arrays).paramaterise(
                _dtag,
                structure_array_refs[_dtag],
                structure_array_refs[dtag],
            ) for _dtag in comparator_datasets}
        )
        alignment_refs = {_dtag: processor.put(alignments[_dtag]) for _dtag in comparator_datasets}
        time_finish_get_alignments = time.time()
        print(f"\t\tGot alignments in: {round(time_finish_get_alignments - time_begin_get_alignments, 2)}")

        # Get the reference frame and save it to the object store
        time_begin_get_frame = time.time()
        reference_frame: DFrame = DFrame(dataset, processor)
        reference_frame_ref = processor.put(reference_frame)
        time_finish_get_frame = time.time()
        print(f"\t\tGot dmaps in: {round(time_finish_get_frame - time_begin_get_frame, 2)}")

        # Get the transforms to apply to the dataset before locally aligning and save them to the object store
        transforms = [
            TruncateReflections(
                comparator_datasets,
                processing_res,
            ),
            SmoothReflections(dataset)
        ]
        transforms_ref = processor.put(transforms)

        # Get the dmaps
        time_begin_get_dmaps = time.time()
        dmaps_dict = processor.process_dict(
            {
                _dtag: Partial(SparseDMapStream.parallel_load).paramaterise(
                    dataset_refs[_dtag],
                    alignment_refs[_dtag],
                    transforms_ref,
                    reference_frame_ref
                )
                for _dtag
                in comparator_datasets
            }
        )
        dmaps = np.vstack([_dmap.data.reshape((1, -1)) for _dtag, _dmap in dmaps_dict.items()])
        time_finish_get_dmaps = time.time()
        print(f"\t\tGot dmaps in: {round(time_finish_get_dmaps - time_begin_get_dmaps, 2)}")
        dtag_array = np.array([_dtag for _dtag in comparator_datasets])

        # Get the dataset dmap
        dtag_index = np.argwhere(dtag_array == dtag)
        print(f"Dtag index: {dtag_index}")
        dataset_dmap_array = dmaps[dtag_index[0][0], :]
        xmap_grid = reference_frame.unmask(SparseDMap(dataset_dmap_array))

        # Get the masked grid of the structure
        model_grid = get_model_map(dataset.structure.structure, xmap_grid)

        # Get the Comparator sets that define the models to try
        time_begin_get_characterization_sets = time.time()
        characterization_sets: Dict[int, Dict[str, DatasetInterface]] = get_characterization_sets(
            dtag,
            comparator_datasets,
            dmaps,
            reference_frame,
            # CharacterizationGaussianMixture(
            #     n_components=min(20, int(len(comparator_datasets) / 25)),
            #     covariance_type="full",
            # ),
            CharacterizationNN()
        )
        time_finish_get_characterization_sets = time.time()
        print(
            f"\t\tGot characterization sets in: {round(time_finish_get_characterization_sets - time_begin_get_characterization_sets, 2)}")

        # Filter the models which are clearly poor descriptions of the density
        # In theory this step could result in the exclusion of a ground state model which provided good contrast
        # for a ligand binding in one part of the protein but fit poorly to say a large disordered region
        time_begin_process_models = time.time()
        model_events = {}
        model_means = {}
        model_zs = {}
        characterization_set_masks = {}
        for model_number, characterization_set in characterization_sets.items():

            # Get the characterization set dmaps
            characterization_set_mask_list = []
            for _dtag in comparator_datasets:
                if _dtag in characterization_set:
                    characterization_set_mask_list.append(True)
                else:
                    characterization_set_mask_list.append(False)
            characterization_set_mask = np.array(characterization_set_mask_list)
            characterization_set_masks[model_number] = characterization_set_mask


        model_scores = {}
        for model_number in characterization_set_masks:
            characterization_set_dmaps_array = dmaps[characterization_set_masks[model_number], :]
            mean, std, z = PointwiseNormal()(
                dataset_dmap_array,
                characterization_set_dmaps_array
            )
            inner_mask_zmap = z[reference_frame.mask.indicies_sparse_inner_atomic]
            percentage_z_2 = float(np.sum(np.abs(inner_mask_zmap) > 2)) / inner_mask_zmap.size
            print(f"Model number: {model_number}: z > 2: {percentage_z_2}")
            print(f"Model number: {model_number}: {np.min(std)} {np.mean(std)}  {np.max(std)} {np.std(std)}")
            print(f"Model number: {np.quantile(z, (0.8, 0.85, 0.9, 0.95))}")

            mean_grid = reference_frame.unmask(SparseDMap(mean))
            mean_grid_array = np.array(mean_grid, copy=False)
            print(mean_grid_array.shape)
            print(np.nonzero(mean_grid_array == 0))
            mask_array = np.zeros(mean_grid_array.shape)
            mask_array[reference_frame.mask.indicies] = 1
            non_zero = np.nonzero((mean_grid_array == 0) & (mask_array == 1))
            for j in range(non_zero[0].size):
                print(f"{non_zero[0][j]} : {non_zero[1][j]} : {non_zero[2][j]}")
            # print(np.)

            model_scores[model_number] = percentage_z_2

        models_to_process = []
        _l = 0
        for model_number in sorted(model_scores, key=lambda _model_number: model_scores[_model_number]):
            if (_l < 3) or (model_scores[model_number] < 0.2):
                models_to_process.append(model_number)
                _l = _l + 1

        print(f"Models to process: {models_to_process}")


        # Process the models: calculating statistical maps; using them to locate events; filtering, scoring and re-
        # filtering those events and returning those events and unpacking them
        processed_models = {
            model_number: Partial(process_model).paramaterise(
                dataset.ligand_files,
                model_number,
                dataset_dmap_array,
                dmaps[characterization_set_masks[model_number], :],
                reference_frame,
                reference_frame.mask_grid(model_grid).data,
                score
            )()
            for model_number
            in models_to_process
        }


        for model_number, result in processed_models.items():
            if result[0] is not None:
                model_events[model_number] = result[0]
                model_means[model_number] = result[1]
                model_zs[model_number] = result[2]

        time_finish_process_models = time.time()
        print(f"\t\tProcessed all models in: {round(time_finish_process_models - time_begin_process_models, 2)}")
        model_events = {model_number: events for model_number, events in model_events.items() if len(events) > 0}
        if len(model_events) == 0:
            print(f"NO EVENTS FOR DATASET {dtag}: SKIPPING REST OF PROCESSING!")
            continue

        # Select a model and the events to output event maps for and to autobuild
        # This step can be dangerous in that events with high multiplity (for example due to NCS) could be filtered
        selected_model_num, selected_events = select_model(model_events)
        print(f"Selected model number: {selected_model_num}")
        selected_model_events = {(dtag, _event_idx): selected_events[_event_idx] for _event_idx in selected_events}
        top_selected_model_events = {
            event_id: selected_model_events[event_id]
            for event_id
            in list(
                sorted(
                    selected_model_events,
                    key=lambda _event_id: selected_model_events[_event_id].score,
                    reverse=True,
                )
            )[:3]
        }

        for event_id, event in top_selected_model_events.items():
            pandda_events[event_id] = event

        # Output models
        output_models(fs, characterization_sets, selected_model_num)

        # Output events
        output_events(fs, top_selected_model_events)

        # Output event maps and model maps
        time_begin_output_maps = time.time()
        output_maps(
            dtag,
            fs,
            top_selected_model_events,
            dataset_dmap_array,
            model_means[selected_model_num],
            model_zs[selected_model_num],
            reference_frame,
            processing_res
        )
        time_finish_output_maps = time.time()
        print(f"\t\tOutput maps in: {round(time_finish_output_maps - time_begin_output_maps, 2)}")

        time_finish_process_dataset = time.time()
        print(f"\tProcessed dataset in {round(time_finish_process_dataset - time_begin_process_dataset, 2)}")

    time_finish_process_datasets = time.time()
    print(
        f"Processed {len(datasets)} datasets in {round(time_finish_process_datasets - time_begin_process_datasets, 2)}")

    # console.summarise_shells(shell_results, all_events, event_scores)

    # Autobuild the best scoring event for each dataset
    console.start_autobuilding()

    # fs_ref = processor.put(fs)
    time_begin_autobuild = time.time()

    best_events = {}
    for dtag in datasets:
        dtag_events = {_event_id: pandda_events[_event_id] for _event_id in pandda_events if _event_id[0] == dtag}
        if len(dtag_events) == 0:
            continue
        best_dtag_event_id = max(dtag_events, key=lambda _event_id: dtag_events[_event_id].score)
        best_events[best_dtag_event_id] = pandda_events[best_dtag_event_id]

    processor.shutdown()
    processor: ProcessorInterface = ProcessLocalRay(args.local_cpus)
    dataset_refs = {_dtag: processor.put(datasets[_dtag]) for _dtag in datasets}
    fs_ref = processor.put(fs)

    time_autobuild_begin = time.time()

    # best_event_autobuilds: Dict[Tuple[str, int], Dict[str, AutobuildInterface]] = processor.process_dict(
    #     {
    #         _event_id: Partial(autobuild).paramaterise(
    #             _event_id,
    #             dataset_refs[_event_id[0]],
    #             pandda_events[_event_id],
    #             AutobuildPreprocessStructure(),
    #             AutobuildPreprocessDMap(),
    #             # Rhofit(cut=1.0),
    #             AutobuildInbuilt(),
    #             fs_ref
    #         )
    #         for _event_id
    #         in best_events
    #     }
    # )

    event_autobuilds: Dict[Tuple[str, int], Dict[str, AutobuildInterface]] = processor.process_dict(
        {
            _event_id: Partial(autobuild).paramaterise(
                _event_id,
                dataset_refs[_event_id[0]],
                pandda_events[_event_id],
                AutobuildPreprocessStructure(),
                AutobuildPreprocessDMap(),
                # Rhofit(cut=1.0),
                AutobuildInbuilt(),
                fs_ref
            )
            for _event_id
            in pandda_events
        }
    )



    # best_event_autobuilds: Dict[Tuple[str, int], Dict[str, AutobuildInterface]] = {
    #         _event_id: autobuild(
    #             _event_id,
    #             datasets[_event_id[0]],
    #             pandda_events[_event_id],
    #             AutobuildPreprocessStructure(),
    #             AutobuildPreprocessDMap(),
    #             # Rhofit(cut=1.0),
    #             AutobuildInbuilt(),
    #             fs
    #         )
    #         for _event_id
    #         in best_events
    #     }

    time_autobuild_finish = time.time()
    print(f"Autobuilt in: {time_autobuild_finish-time_autobuild_begin}")

    # autobuilds = {}
    # for _event_id in pandda_events:
    #     if _event_id in best_event_autobuilds:
    #         autobuilds[_event_id] = best_event_autobuilds[_event_id]
    #     else:
    #         autobuilds[_event_id] = {ligand_key: AutobuildResult(None, None, None, None, None, None) for ligand_key in
    #                                  datasets[_event_id[0]].ligand_files}
    # time_finish_autobuild = time.time()
    # print(f"Autobuilt {len(best_event_autobuilds)} events in: {round(time_finish_autobuild - time_begin_autobuild, 1)}")

    autobuilds = {}
    for _event_id in pandda_events:
        if _event_id in event_autobuilds:
            autobuilds[_event_id] = event_autobuilds[_event_id]
        else:
            autobuilds[_event_id] = {ligand_key: AutobuildResult(None, None, None, None, None, None) for ligand_key in
                                     datasets[_event_id[0]].ligand_files}
    time_finish_autobuild = time.time()
    print(f"Autobuilt {len(pandda_events)} events in: {round(time_finish_autobuild - time_begin_autobuild, 1)}")
    # console.summarise_autobuilding(autobuild_results)

    # Merge the autobuilds
    merged_build_scores = merge_autobuilds(
        datasets,
        pandda_events,
        autobuilds,
        fs,
        # MergeHighestRSCC(),
        # MergeHighestBuildScore()
        MergeHighestBuildAndEventScore()

    )

    # Get the sites
    sites: Dict[int, Site] = get_sites(
        pandda_events,
        HeirarchicalSiteModel(t=8.0)
    )
    for site_id, site in sites.items():
        print(f"{site_id} : {site.centroid} : {site.event_ids}")

    # rank
    ranking = rank_events(
        pandda_events,
        autobuilds,
        RankHighEventScore(),
        # RankHighBuildScore()
    )
    for event_id in ranking:
        print(f"{event_id} : {round(pandda_events[event_id].score, 2)}")

    # Output tables
    output_tables(pandda_events, ranking, sites, fs)
    time_pandda_finish = time.time()
    print(f"PanDDA ran in: {time_pandda_finish-time_pandda_begin}")


if __name__ == '__main__':
    # Parse Command Line Arguments
    args = PanDDAArgs.from_command_line()

    # Process the PanDDA
    pandda(args)
