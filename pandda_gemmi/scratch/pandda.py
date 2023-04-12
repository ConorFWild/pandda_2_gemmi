import time

import numpy as np

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
)

from pandda_gemmi.scratch.event_model.evaluate import evaluate_model
from pandda_gemmi.scratch.event_model.characterization import get_characterization_sets, CharacterizationGaussianMixture
from pandda_gemmi.scratch.event_model.outlier import PointwiseNormal
from pandda_gemmi.scratch.event_model.cluster import ClusterDensityDBSCAN
from pandda_gemmi.scratch.event_model.score import ScoreCNN, get_model_map
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


def pandda(args: PanDDAArgs):
    # Get the processor
    processor: ProcessorInterface = ProcessLocalRay(args.local_cpus)

    # Get the FS
    fs: PanDDAFSInterface = PanDDAFS(Path(args.data_dirs), Path(args.out_dir))

    # Get the scoring method
    score = ScoreCNN()

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
    for dtag in datasets:
        # if dtag != "JMJD2DA-x427":
        #     continue
        print(f"##### {dtag} #####")
        time_begin_process_dataset = time.time()

        # Get the dataset
        dataset = datasets[dtag]

        # Get the resolution to process at
        dataset_res = dataset.reflections.resolution() + 0.1
        processing_res = max(dataset_res,
                             list(sorted([_dataset.reflections.resolution() for _dataset in datasets.values()]))[
                                 60] + 0.1)

        # Get the comparator datasets
        comparator_datasets: Dict[str, DatasetInterface] = get_comparators(
            datasets,
            [FilterRFree(0.4), FilterSpaceGroup(dataset), FilterResolution(processing_res)]
        )

        # Get the alignments
        alignments: Dict[str, AlignmentInterface] = processor.process_dict(
            {_dtag: Partial(Alignment.from_structure_arrays).paramaterise(
                structure_array_refs[_dtag],
                structure_array_refs[dtag],
            ) for _dtag in comparator_datasets}
        )
        alignment_refs = {_dtag: processor.put(alignments[_dtag]) for _dtag in comparator_datasets}

        # Get the reference frame
        reference_frame: DFrame = DFrame(dataset, processor)
        reference_frame_ref = processor.put(reference_frame)

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
        dmaps = np.vstack([dmap.data.reshape((1, -1)) for dtag, dmap in dmaps_dict.items()])

        # Get the relevant dmaps
        dtag_array = np.array([_dtag for _dtag in comparator_datasets])

        # Get the dataset dmap
        dtag_index = np.argwhere(dtag_array == dtag)
        dataset_dmap_array = dmaps[dtag_index, :]
        xmap_grid = reference_frame.unmask(SparseDMap(dataset_dmap_array))

        #
        model_grid = get_model_map(dataset.structure.structure, xmap_grid)

        # Comparator sets
        characterization_sets: Dict[int, Dict[str, DatasetInterface]] = get_characterization_sets(
            dtag,
            comparator_datasets,
            dmaps,
            reference_frame,
            CharacterizationGaussianMixture(n_components=20, covariance_type="diag"),
        )

        model_events = {}
        model_means = {}
        model_zs = {}
        for model_number, characterization_set in characterization_sets.items():

            # Get the characterization set dmaps
            characterization_set_mask_list = []
            for _dtag in comparator_datasets:
                if _dtag in characterization_set:
                    characterization_set_mask_list.append(True)
                else:
                    characterization_set_mask_list.append(False)
            characterization_set_mask = np.array(characterization_set_mask_list)
            characterization_set_dmaps_array = dmaps[characterization_set_mask, :]

            # Get the statical maps
            mean, std, z = PointwiseNormal()(
                dataset_dmap_array,
                characterization_set_dmaps_array
            )
            model_means[model_number] = mean
            model_zs[model_number] = z

            mean_grid = reference_frame.unmask(SparseDMap(mean))
            z_grid = reference_frame.unmask(SparseDMap(z))

            # Initial
            events = ClusterDensityDBSCAN()(z, reference_frame)
            print(f"Initial events: {len(events)}")

            # Filter the events pre-scoring
            for filter in [FilterSize(reference_frame, min_size=5.0), FilterCluster(5.0), ]:
                events = filter(events)

            print(f"After filer size and cluster: {len(events)}")

            if len(events) == 0:
                continue

            # Score the events
            events = score(events, xmap_grid, mean_grid, z_grid, model_grid)

            # Filter the events post-scoring
            for filter in [FilterScore(0.1), FilterLocallyHighestScoring(10.0)]:
                events = filter(events)
                print(f"After filter score: {len(events)}")

            if len(events) == 0:
                continue

            print(f"Events: {[round(x, 2) for x in sorted([event.score for event in events.values()])]}")

            model_events[model_number] = events

        model_events = {model_number: events for model_number, events in model_events.items() if len(events) > 0}
        if len(model_events) == 0:
            continue

        # Select a model
        selected_model_num, selected_events = select_model(model_events)
        model_events = {(dtag, _event_idx): selected_events[_event_idx] for _event_idx in selected_events}
        for event_id, event in model_events.items():
            pandda_events[event_id] = event

        # Output models
        output_models(fs, characterization_sets, selected_model_num)

        # Output events
        output_events(fs, model_events)

        # Output event maps and model maps
        output_maps(
            dtag,
            fs,
            {(dtag, _event_idx): selected_events[_event_idx] for _event_idx in selected_events},
            dataset_dmap_array,
            model_means[selected_model_num],
            model_zs[selected_model_num],
            reference_frame,
        )

        time_finish_process_dataset = time.time()
        print(f"\tProcessed dataset in {round(time_finish_process_dataset-time_begin_process_dataset, 2)}")

    time_finish_process_datasets = time.time()
    print(f"Processed {len(datasets)} datasets in {round(time_finish_process_datasets - time_begin_process_datasets, 2)}")

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
            autobuilds[_event_id] = AutobuildResult(None, None, None, None, None, None)
    time_finish_autobuild = time.time()
    print(f"Autobuilt {len(best_event_autobuilds)} events in: {round(time_finish_autobuild-time_begin_autobuild, 1)}")

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
