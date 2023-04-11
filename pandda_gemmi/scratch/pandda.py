from pandda_gemmi.scratch.interfaces import *

from pandda_gemmi.scratch.fs import PanDDAFS
from pandda_gemmi.scratch.dataset import Dataset
from pandda_gemmi.scratch.dmaps import DMap, SparseDMap, SparseDMapStream
from pandda_gemmi.scratch.alignment import Alignment, DFrame
from pandda_gemmi.scratch.processor import ProcessLocalRay

from pandda_gemmi.scratch.event_model.model import EventModel
from pandda_gemmi.scratch.event_model.select import SelectNN
from pandda_gemmi.scratch.event_model.outlier import PointwiseNormal
from pandda_gemmi.scratch.event_model.cluster import ClusterDensityAgg
from pandda_gemmi.scratch.event_model.score import ScoreCNN
from pandda_gemmi.scratch.event_model.filter import FilterSize, FilterScore

from pandda_gemmi.scratch.site_model import SiteModel, ClusterSitesAgg, Site

from pandda_gemmi.scratch.autobuild import Autobuild
from pandda_gemmi.scratch.autobuild.rhofit import Rhofit
from pandda_gemmi.scratch.autobuild.merge import MergeHighestRSCC




def pandda():
    # Get the processor
    processor: Processor

    # Get the FS
    fs: FS

    # Get the datasets
    datasets: Dict[str, Dataset]
    dataset_refs = {_dtag: processor.put(datasets[_dtag]) for _dtag in datasets}
    structure_array_refs = {_dtag: processor.put(StructureArray.from_structure(datasets[_dtag].structure)) for _dtag in
                            datasets}

    # Get the

    # Process each dataset
    pandda_events = {}
    for dtag in datasets:
        # Get the dataset
        dataset = datasets[dtag]

        # Get the comparator datasets
        comparator_datasets: Dict[str, DatasetInterface] = GetComparators(
            [FilterRFree, FilterSpaceGroup, FilterResolution]
        )()

        # Get the alignments
        alignments: Dict[str, AlignmentInterface] = processor.process_dict(
        {_dtag: Partial(Alignment.from_structure_arrays).paramaterise(
            structure_array_refs[_dtag],
            structure_array_refs[dtag],
        ) for _dtag in comparator_datasets}
    )

        # Get the reference frame
        reference_frame: DFrame = DFrame(dataset, processor)

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
                in datasets_resolution}
        )

        # Get comparators
        # comparator_sets: Dict[int, List[str]] =

        # Comparator sets
        comparator_sets = SelectGuassianMixture(dtag, datasets, dmaps)

        # Get the models
        event_model = EventModel(
            PointwiseNormal,
            ClusterDensityDBSCAN,
            ScoreCNN,
            [FilterSize, FilterCluster],
            processor
        )
        models: Dict[int, EDModel] = {event_model(char_set) for char_set in char_sets}

        # Evaluate the models against the dataset
        events: Dict[int, List[EventInterface]] = {model_num: model.evaluate() for model in models}

        # Select a model
        selected_model, selected_events = select_model(events)
        pandda_events[dtag] = selected_events

        # Output models
        output_models()

        # Output events
        output_events()

        # Output event maps and model maps
        output_maps()


    # Autobuild
    autobuilds = Autobuild(
        Rhofit,
        MergeHighestRSCC
    )
    autobuilds = processor.process_dict(
        {
            _event_id: Partial(autobuild).paramaterise(

            )
            for _event_id
            in pandda_events
        }
    )

    # Get the sites
    sites: Dict[int, Site] = SiteModel().evaluate()

    # rank
    ranking = RankScore()(dataset_events)

    # Output tables
    output_tables(dataset_events)
