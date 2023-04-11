from pandda_gemmi.scratch.interfaces import *

from pandda_gemmi.args import PanDDAArgs

from pandda_gemmi.scratch.fs import PanDDAFS
from pandda_gemmi.scratch.dataset import XRayDataset, StructureArray
from pandda_gemmi.scratch.dmaps import (
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
from pandda_gemmi.scratch.event_model.score import ScoreCNN
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
        processing_res = dataset.reflections.resolution() + 0.1

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
        dmaps = processor.process_dict(
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

        # Comparator sets
        characterization_sets: Dict[int, Dict[str, DatasetInterface]] = get_characterization_sets(
            dtag,
            datasets,
            dmaps,
            CharacterizationGaussianMixture(),
        )

        # Get the models
        statistical_model = PointwiseNormal()
        cluster_density = ClusterDensityDBSCAN()
        pre_score_filters = [FilterSize(), FilterCluster(), ]
        scoring = ScoreCNN()
        post_score_filters = [FilterScore(0.1), ]

        # Evaluate the models against the dataset
        model_events: Dict[int, Dict[int, EventInterface]] = {
            model_number: evaluate_model(
                characterization_set,
                statistical_model,
                cluster_density,
                pre_score_filters,
                scoring,
                post_score_filters
            )
            for model_number, characterization_set
            in characterization_sets.items()
        }

        # Select a model
        selected_model, selected_events = select_model(model_events)
        pandda_events[dtag] = {(dtag, _event_idx): selected_events[_event_idx] for _event_idx in selected_events}

        # Output models
        output_models(characterization_sets, selected_model)

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
