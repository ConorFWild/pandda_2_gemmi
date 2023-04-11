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
from pandda_gemmi.scratch.event_model.filter import FilterSize, FilterCluster, FilterScore
from pandda_gemmi.scratch.event_model.select import select_model
from pandda_gemmi.scratch.event_model.output import output_models, output_events, output_maps

from pandda_gemmi.scratch.site_model import SiteModel, ClusterSites, Site, get_sites

from pandda_gemmi.scratch.autobuild import autobuild
from pandda_gemmi.scratch.autobuild.rhofit import Rhofit
from pandda_gemmi.scratch.autobuild.merge import MergeHighestRSCC

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
    for dtag in datasets:
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
        dtag_array = np.array([_dtag for _dtag in datasets])

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
        for model_number, characterization_set in characterization_sets.items():


            # Get the characterization set dmaps
            characterization_set_mask_list = []
            for _dtag in datasets:
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
            mean_grid = reference_frame.unmask(SparseDMap(mean))
            z_grid = reference_frame.unmask(SparseDMap(z))

            # Initial
            events = ClusterDensityDBSCAN()(z, reference_frame)

            # Filter the events pre-scoring
            for filter in [FilterSize(reference_frame, min_size=5.0), FilterCluster(5.0), ]:
                events = filter(events)

            # Score the events
            events = score(events, xmap_grid, mean_grid, z_grid, model_grid)

            # Filter the events post-scoring
            for filter in [FilterScore(0.1), ]:
                events = filter(events)

            model_events[model_number] = events

        # Select a model
        selected_model_num, selected_events = select_model(model_events)
        pandda_events[dtag] = {(dtag, _event_idx): selected_events[_event_idx] for _event_idx in selected_events}

        # Output models
        output_models(characterization_sets, selected_model_num)

        # Output events
        output_events(model_events)

        # Output event maps and model maps
        output_maps(selected_events)

    # Autobuild
    autobuilds: Dict[Tuple[str, int], AutobuildInterface] = processor.process_dict(
        {
            _event_id: Partial(autobuild).paramaterise(
                pandda_events[_event_id],
                Rhofit(),
            )
            for _event_id
            in pandda_events
        }
    )

    # Get the sites
    sites: Dict[int, Site] = get_sites(
        pandda_events,
        SiteModel()
    )

    # rank
    ranking = rank_events(
        pandda_events,
        autobuilds,
        RankHighScore,
    )

    # Output tables
    output_tables(pandda_events)


if __name__ == '__main__':
    # Parse Command Line Arguments
    args = PanDDAArgs.from_command_line()

    # Process the PanDDA
    pandda(args)
