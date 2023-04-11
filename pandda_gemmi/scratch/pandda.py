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

    # Get the

    # Process each dataset
    dataset_events = {}
    for dtag in datasets:
        # Get the alignments
        alignments: Alignment

        # Get the reference frame
        reference_frame: DFrame

        # Get the dmaps
        dmaps: DMapStream

        # Get comparators
        comparators: Dict[int, List[str]] = GetComparators(
            [FilterRFree, FilterSpaceGroup]
        )()

        #
        dataset, dmap, other_dmaps = datasets[dtag], dmaps[dtag], {_dtag: _dmap for _dtag, _dmap in dmaps.items() if
                                                                   _dtag != dtag}

        # Comparator sets
        comparator_sets = SelectNN(dtag, datasets, dmaps)

        # Get the models
        event_model = EventModel(
            SelectNN,
            PointwiseNormal,
            ClusterDensityAgg,
            ScoreCNN,
            [FilterSize, FilterScore],
        )
        models: Dict[int, EDModel] = get_event_models(dataset, dmap, other_dmaps, )

        # Evaluate the models against the dataset
        events: Dict[int, List[Event]] = {model_num: model.evaluate() for model in models}

        # Select a model
        selected_model, dataset_events[dtag] = select_model(events)

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

    # Get the sites
    sites: Dict[int, Site] = SiteModel().evaluate()

    # rank
    ranking = RankScore()(dataset_events)

    # Output tables
    output_tables(dataset_events)
