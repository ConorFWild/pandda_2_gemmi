import time

import numpy as np
from sklearn.decomposition import FastICA, FactorAnalysis, NMF

from pandda_gemmi.scratch.interfaces import *

from pandda_gemmi.args import PanDDAArgs

from pandda_gemmi.scratch.fs import PanDDAFS
from pandda_gemmi.scratch.dataset import XRayDataset, StructureArray
from pandda_gemmi.scratch.dmaps import (
    SparseDMap,
    SparseDMapStream,
    TruncateReflections,
    SmoothReflections,
)
from pandda_gemmi.scratch.alignment import Alignment, DFrame
from pandda_gemmi.scratch.processor import ProcessLocalRay, Partial

from pandda_gemmi.scratch.comparators import (
    get_comparators,
    FilterRFree,
    FilterSpaceGroup,
    FilterResolution,
    FilterCompatibleStructures
)
from pandda_gemmi.scratch import constants
from pandda_gemmi.scratch.dmaps import save_dmap
from pandda_gemmi.scratch.event_model.evaluate import evaluate_model
from pandda_gemmi.scratch.event_model.characterization import get_characterization_sets, CharacterizationGaussianMixture
from pandda_gemmi.scratch.event_model.outlier import PointwiseNormal
from pandda_gemmi.scratch.event_model.cluster import ClusterDensityDBSCAN
from pandda_gemmi.scratch.event_model.score import ScoreCNN, get_model_map, ScoreCNNLigand
from pandda_gemmi.scratch.event_model.filter import FilterSize, FilterCluster, FilterScore, FilterLocallyHighestScoring
from pandda_gemmi.scratch.event_model.select import select_model
from pandda_gemmi.scratch.event_model.output import output_models, output_events, output_maps

from pandda_gemmi.scratch.site_model import HeirarchicalSiteModel, ClusterSites, Site, get_sites

from pandda_gemmi.scratch.autobuild import autobuild, AutobuildResult
from pandda_gemmi.scratch.autobuild.rhofit import Rhofit
from pandda_gemmi.scratch.autobuild.merge import merge_autobuilds, MergeHighestRSCC
from pandda_gemmi.scratch.autobuild.preprocess_structure import AutobuildPreprocessStructure
from pandda_gemmi.scratch.autobuild.preprocess_dmap import AutobuildPreprocessDMap

from pandda_gemmi.scratch.ranking import rank_events, RankHighScore

from pandda_gemmi.scratch.tables import output_tables


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
    # print([dataset_dmap_array.shape, reference_frame.mask.indicies_sparse_inner_atomic.shape])
    inner_mask_xmap = dataset_dmap_array[reference_frame.mask.indicies_sparse_inner_atomic]
    # median = np.median(inner_mask_xmap)
    median = np.median(mean[reference_frame.mask.indicies_sparse_inner_atomic])

    # median = np.quantile(inner_mask_xmap, 0.05)
    # median = np.quantile(
    #     inner_mask_xmap,
    #     np.linspace(0.05, 0.95, 10)
    # )

    # print(f"Median is: {median}")
    model_grid = reference_frame.unmask(SparseDMap(model_map))

    inner_mask_zmap = z[reference_frame.mask.indicies_sparse_inner_atomic]
    percentage_z_2 = float(np.sum(np.abs(inner_mask_zmap) > 2)) / inner_mask_zmap.size
    print(f"Model number: {model_number}: z > 2: {percentage_z_2}")

    # Initial
    events = ClusterDensityDBSCAN()(z, reference_frame)
    print(f"Initial events: {len(events)}")

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
        FilterLocallyHighestScoring(8.0),
    ]:
        events = filter(events)
    # print(f"After filter score: {len(events)}")

    if len(events) == 0:
        return None, None, None

    # print(f"Events: {[round(x, 2) for x in sorted([event.score for event in events.values()])]}")

    # model_events[model_number] = events

    return events, mean, z


def pandda(args: PanDDAArgs):
    # Get the processor
    processor: ProcessorInterface = ProcessLocalRay(args.local_cpus)

    # Get the FS
    fs: PanDDAFSInterface = PanDDAFS(Path(args.data_dirs), Path(args.out_dir))

    # Get the scoring method
    score = ScoreCNNLigand()
    score_ref = processor.put(score)

    # Get the datasets
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

    # Process each dataset
    pandda_events = {}
    time_begin_process_datasets = time.time()
    _k = 0
    for dtag in datasets:
        # _k += 1
        # if _k > 3:
        #     continue
        # if dtag != "JMJD2DA-x427":
        #     continue

        # if dtag != "JMJD2DA-x348":
        #     continue

        if dtag != "Zika_NS3A-A0340":
            continue
        print(f"##### {dtag} #####")
        time_begin_process_dataset = time.time()

        # Get the dataset
        dataset = datasets[dtag]

        if dtag == "Zika_NS3A-A0340":
            print(dataset.ligand_files)

        # Get the resolution to process at
        dataset_res = dataset.reflections.resolution()
        # processing_res = max(dataset_res,
        #                      list(
        #                          sorted(
        #                              [_dataset.reflections.resolution() for _dataset in datasets.values()]
        #                          )
        #                      )[60] + 0.1)
        print(
            f"Dataset resolution is: {dataset.reflections.resolution()}")
        # print(f"Dataset rfree is: {dataset.structure.rfree()}")

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

        processing_res = max(
                    [_dataset.reflections.resolution() for _dataset in comparator_datasets.values()]
            )
        print(f"Processing res is: {processing_res}")
        print(f"Number of comparator datasets: {len(comparator_datasets)}")
        # if len(comparator_datasets) < 60:
        #     _h = 0
        #     for _dtag in sorted(datasets, key=lambda __dtag: datasets[__dtag].reflections.resolution()):
        #         if _h >60:
        #             continue
        #         comparator_datasets[_dtag] = datasets[_dtag]
        #         _h = _h + 1



        if dtag not in comparator_datasets:
            comparator_datasets[dtag] = dataset

        # Get the alignments
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

        # Get the reference frame
        time_begin_get_frame = time.time()
        reference_frame: DFrame = DFrame(dataset, processor)
        reference_frame_ref = processor.put(reference_frame)
        time_finish_get_frame = time.time()
        print(f"\t\tGot dmaps in: {round(time_finish_get_frame - time_begin_get_frame, 2)}")

        # Get the transforms
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

        # Get the relevant dmaps
        dtag_array = np.array([_dtag for _dtag in comparator_datasets])

        # Get the dataset dmap
        dtag_index = np.argwhere(dtag_array == dtag)
        print(f"Dtag index: {dtag_index}")
        dataset_dmap_array = dmaps[dtag_index[0][0], :]
        xmap_grid = reference_frame.unmask(SparseDMap(dataset_dmap_array))

        #
        model_grid = get_model_map(dataset.structure.structure, xmap_grid)
        model_map_ref = processor.put(reference_frame.mask_grid(model_grid).data)

        # Comparator sets
        time_begin_get_characterization_sets = time.time()
        characterization_sets: Dict[int, Dict[str, DatasetInterface]] = get_characterization_sets(
            dtag,
            comparator_datasets,
            dmaps,
            reference_frame,
            CharacterizationGaussianMixture(n_components=min(20, int(len(comparator_datasets) / 25)),
                                            covariance_type="full"),
        )
        time_finish_get_characterization_sets = time.time()
        print(
            f"\t\tGot characterization sets in: {round(time_finish_get_characterization_sets - time_begin_get_characterization_sets, 2)}")

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
            # characterization_set_dmaps_array = dmaps[characterization_set_mask, :]
        # for model_number, characterization_set in characterization_sets.items():
        #
        #     # Get the characterization set dmaps
        #     characterization_set_mask_list = []
        #     for _dtag in comparator_datasets:
        #         if _dtag in characterization_set:
        #             characterization_set_mask_list.append(True)
        #         else:
        #             characterization_set_mask_list.append(False)
        #     characterization_set_mask = np.array(characterization_set_mask_list)
        #     characterization_set_dmaps_array = dmaps[characterization_set_mask, :]
        #
        #     # Get the statical maps
        #     mean, std, z = PointwiseNormal()(
        #         dataset_dmap_array,
        #         characterization_set_dmaps_array
        #     )
        #     model_means[model_number] = mean
        #     model_zs[model_number] = z
        #
        #     mean_grid = reference_frame.unmask(SparseDMap(mean))
        #     z_grid = reference_frame.unmask(SparseDMap(z))
        #
        #     # Initial
        #     events = ClusterDensityDBSCAN()(z, reference_frame)
        #     print(f"Initial events: {len(events)}")
        #
        #     # Filter the events pre-scoring
        #     for filter in [FilterSize(reference_frame, min_size=5.0), FilterCluster(5.0), ]:
        #         events = filter(events)
        #
        #     print(f"After filer size and cluster: {len(events)}")
        #
        #     if len(events) == 0:
        #         continue
        #
        #     # Score the events
        #     time_begin_score_events = time.time()
        #     events = score(events, xmap_grid, mean_grid, z_grid, model_grid)
        #     time_finish_score_events = time.time()
        #     print(f"\t\t\tScored events in: {round(time_finish_score_events - time_begin_score_events, 2)}")
        #
        #     # Filter the events post-scoring
        #     for filter in [FilterScore(0.1), FilterLocallyHighestScoring(10.0)]:
        #         events = filter(events)
        #     print(f"After filter score: {len(events)}")
        #
        #     if len(events) == 0:
        #         continue
        #
        #     print(f"Events: {[round(x, 2) for x in sorted([event.score for event in events.values()])]}")
        #
        #     model_events[model_number] = events

        model_scores = {}
        for model_number in characterization_set_masks:
            characterization_set_dmaps_array = dmaps[characterization_set_masks[model_number], :]
            mean, std, z = PointwiseNormal()(
                dataset_dmap_array,
                characterization_set_dmaps_array
            )

            # mat = np.vstack(
            #     [
            #         mean.reshape((1, -1)),
            #         dataset_dmap_array.reshape((1, -1))
            #     ])
            #
            # mat_nonzero = np.copy(mat)
            # mat_nonzero[mat_nonzero<0] = 0.0
            #
            # transformer = NMF(n_components=2, max_iter=1000)
            # transformed = transformer.fit_transform(mat_nonzero)
            # print(f"NMF transformed shape: {transformed.shape}")
            # print(transformed)
            # print(f"Fraction event: {transformed[1, 1] / transformed[0, 1]}")
            #
            # components = transformer.components_
            #
            # signal_1 = components[0, :].flatten()
            # signal_1_scaled = (signal_1 - np.mean(signal_1)) / np.std(signal_1)
            # save_dmap(
            #     reference_frame.unmask(SparseDMap(signal_1_scaled)),
            #     fs.output.processed_datasets[dtag] / f"model_{model_number}_NMF_0.ccp4"
            # )
            #
            # signal_2 = components[1, :].flatten()
            # signal_2_scaled = (signal_2 - np.mean(signal_2)) / np.std(signal_2)
            # save_dmap(
            #     reference_frame.unmask(SparseDMap(signal_2_scaled)),
            #     fs.output.processed_datasets[dtag] / f"model_{model_number}_NMF_1.ccp4"
            # )
            #
            # transformer = FactorAnalysis(n_components=2)
            # transformed = transformer.fit_transform(mat)
            # print(f"FA transformed shape: {transformed.shape}")
            # print(transformed)
            # components = transformer.components_
            #
            # signal_1 = components[0, :].flatten()
            # signal_1_scaled = (signal_1 - np.mean(signal_1)) / np.std(signal_1)
            # save_dmap(
            #     reference_frame.unmask(SparseDMap(signal_1_scaled)),
            #     fs.output.processed_datasets[dtag] / f"model_{model_number}_fa_0.ccp4"
            # )
            #
            # signal_2 = components[1, :].flatten()
            # signal_2_scaled = (signal_2 - np.mean(signal_2)) / np.std(signal_2)
            # save_dmap(
            #     reference_frame.unmask(SparseDMap(signal_2_scaled)),
            #     fs.output.processed_datasets[dtag] / f"model_{model_number}_fa_1.ccp4"
            # )
            #
            #
            #
            # ica = FastICA(n_components=2)
            # S_ = ica.fit_transform(
            #     np.vstack(
            #         [
            #             mean.reshape((1, -1)),
            #             dataset_dmap_array.reshape((1, -1))
            #         ]).T)
            # A_ = ica.mixing_
            # print(f"MIXING:")
            # print(A_)
            #
            # signal_1 = S_[:,0].flatten()
            # signal_1_scaled = (signal_1 - np.mean(signal_1)) / np.std(signal_1)
            # save_dmap(
            #     reference_frame.unmask(SparseDMap(signal_1_scaled)),
            #     fs.output.processed_datasets[dtag] / f"model_{model_number}_ica_0.ccp4"
            # )
            #
            # signal_2 = S_[:,1].flatten()
            # signal_2_scaled = (signal_2 - np.mean(signal_2)) / np.std(signal_2)
            # save_dmap(
            #     reference_frame.unmask(SparseDMap(signal_2_scaled)),
            #     fs.output.processed_datasets[dtag] / f"model_{model_number}_ica_1.ccp4"
            # )

            mean_grid = reference_frame.unmask(SparseDMap(mean))
            z_grid = reference_frame.unmask(SparseDMap(z))
            xmap_grid = reference_frame.unmask(SparseDMap(dataset_dmap_array))
            # print([dataset_dmap_array.shape, reference_frame.mask.indicies_sparse_inner_atomic.shape])
            inner_mask_xmap = dataset_dmap_array[reference_frame.mask.indicies_sparse_inner_atomic]
            median = np.median(inner_mask_xmap)
            # print(f"Median is: {median}")
            model_map = reference_frame.mask_grid(model_grid).data
            model_grid = reference_frame.unmask(SparseDMap(model_map))

            inner_mask_zmap = z[reference_frame.mask.indicies_sparse_inner_atomic]
            percentage_z_2 = float(np.sum(np.abs(inner_mask_zmap) > 2)) / inner_mask_zmap.size
            print(f"Model number: {model_number}: z > 2: {percentage_z_2}")
            print(f"Model number: {model_number}: {np.min(std)} {np.mean(std)}  {np.max(std)} {np.std(std)}")

            model_scores[model_number] = percentage_z_2

        models_to_process = []
        _l = 0
        for model_number in sorted(model_scores, key=lambda _model_number: model_scores[_model_number]):
            if (_l < 2) or (model_scores[model_number] < 0.2):
                models_to_process.append(model_number)
                _l = _l + 1

        # processed_models = processor.process_dict(
        #     {
        #         model_number: Partial(process_model).paramaterise(
        #             model_number,
        #             dataset_dmap_array,
        #             dmaps[characterization_set_masks[model_number], :],
        #             reference_frame,
        #             model_map_ref,
        #             score_ref
        #         )
        #         for model_number
        #         in models_to_process
        #     }
        # )

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

            zmap_grid = reference_frame.unmask(SparseDMap(result[2]))
            save_dmap(zmap_grid, fs.output.processed_datasets[dtag] / f"{model_number}_z.ccp4")

        time_finish_process_models = time.time()
        print(f"\t\tProcessed all models in: {round(time_finish_process_models - time_begin_process_models, 2)}")
        # exit()
        model_events = {model_number: events for model_number, events in model_events.items() if len(events) > 0}
        if len(model_events) == 0:
            print(f"NO EVENTS FOR DATASET {dtag}: SKIPPING REST OF PROCESSING!")
            continue

        # Select a model
        selected_model_num, selected_events = select_model(model_events)
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
            )[:10]
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
            # {(dtag, _event_idx): top_selected_model_events[_event_idx] for _event_idx in top_selected_model_events},
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

    # Autobuild the best scoring event for each dataset
    fs_ref = processor.put(fs)
    time_begin_autobuild = time.time()

    best_events = {}
    for dtag in datasets:
        dtag_events = {_event_id: pandda_events[_event_id] for _event_id in pandda_events if _event_id[0] == dtag}
        if len(dtag_events) == 0:
            continue
        best_dtag_event_id = max(dtag_events, key=lambda _event_id: dtag_events[_event_id].score)
        best_events[best_dtag_event_id] = pandda_events[best_dtag_event_id]
    best_event_autobuilds: Dict[Tuple[str, int], Dict[str, AutobuildInterface]] = processor.process_dict(
        {
            _event_id: Partial(autobuild).paramaterise(
                _event_id,
                dataset_refs[_event_id[0]],
                pandda_events[_event_id],
                AutobuildPreprocessStructure(),
                AutobuildPreprocessDMap(),
                Rhofit(cut=1.0),
                fs_ref
            )
            for _event_id
            in best_events
        }
    )
    autobuilds = {}
    for _event_id in pandda_events:
        if _event_id in best_event_autobuilds:
            autobuilds[_event_id] = best_event_autobuilds[_event_id]
        else:
            autobuilds[_event_id] = {ligand_key: AutobuildResult(None, None, None, None, None, None) for ligand_key in
                                     datasets[_event_id[0]].ligand_files}
    time_finish_autobuild = time.time()
    print(f"Autobuilt {len(best_event_autobuilds)} events in: {round(time_finish_autobuild - time_begin_autobuild, 1)}")

    # Merge the autobuilds
    merge_autobuilds(
        datasets,
        pandda_events,
        autobuilds,
        fs,
        MergeHighestRSCC(),
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
        RankHighScore(),
    )
    for event_id in ranking:
        print(f"{event_id} : {round(pandda_events[event_id].score, 2)}")

    # Output tables
    output_tables(pandda_events, ranking, sites, fs)


if __name__ == '__main__':
    # Parse Command Line Arguments
    args = PanDDAArgs.from_command_line()

    # Process the PanDDA
    pandda(args)
