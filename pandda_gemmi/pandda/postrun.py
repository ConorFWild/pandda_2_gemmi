import time

import pandas as pd

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except ImportError:
    print('No sklearn-express available!')

from pandda_gemmi.interfaces import *
from pandda_gemmi import constants
from pandda_gemmi.site_model import HeirarchicalSiteModel, Site, get_sites
from pandda_gemmi.autobuild.merge import merge_autobuilds, MergeHighestBuildScore
from pandda_gemmi.ranking import rank_events, RankHighEventScoreBySite
from pandda_gemmi.tables import output_tables
from pandda_gemmi import serialize
from pandda_gemmi.event_model.event import Event


from pandda_gemmi.metrics import get_hit_in_site_probabilities

def postrun(
        args,
        fs,
        console,
        datasets,
        pandda_events,
        autobuilds,
        datasets_to_process,
        event_score_quantiles,
        time_pandda_begin
):
    # TODO: Log properly
    # print(
    #     f"Processed {len(datasets)} datasets in {round(time_finish_process_datasets - time_begin_process_datasets, 2)}")

    # Get existing site and event data (if it exists)
    inspect_table_file = fs.output.analyses_dir / constants.PANDDA_INSPECT_EVENTS_PATH
    inspect_sites_file = fs.output.analyses_dir / constants.PANDDA_INSPECT_SITES_PATH
    if inspect_table_file.exists():
        inspect_events_table = pd.read_csv(inspect_table_file)
        inspect_sites_table = pd.read_csv(inspect_sites_file)
        existing_events = {(_row['dtag'], int(_row['event_idx'])): _row for _idx, _row in inspect_events_table.iterrows()}
        existing_sites = {
            _row['site_idx']: _row
            for _idx, _row
            in inspect_sites_table.iterrows()
        }
    else:
        existing_events = None
        existing_sites = None

    # Autobuild the best scoring event for each dataset
    console.start_autobuilding()

    autobuild_yaml_path = fs.output.path / "autobuilds.yaml"
    if autobuild_yaml_path.exists():
        autobuilds = serialize.unserialize_autobuilds(autobuild_yaml_path)
    else:
        # Merge the autobuilds into PanDDA output models
        if args.use_ligand_data & args.autobuild:
            merged_build_scores = merge_autobuilds(
                datasets,
                pandda_events,
                autobuilds,
                fs,
                MergeHighestBuildScore()
            )

        #
        console.processed_autobuilds(autobuilds)

    # Get the sites
    sites: Dict[int, Site] = get_sites(
        datasets,
        pandda_events,
        datasets_to_process[
            min(
                datasets_to_process,
                key=lambda _dtag: datasets_to_process[_dtag].reflections.resolution()
            )
        ],
        HeirarchicalSiteModel(t=args.max_site_distance_cutoff, debug=args.debug),
        existing_events,
        existing_sites
    )
    # TODO: Log properly
    # print("Sites")
    # for site_id, site in sites.items():
    #     print(f"{site_id} : {site.centroid} : {site.event_ids}")

    # Rank the events for display in PanDDA inspect
    ranking = rank_events(
        pandda_events,
        sites,
        autobuilds,
        RankHighEventScoreBySite(),
    )
    # for event_id in ranking:
    #     print(f"{event_id} : {round(pandda_events[event_id].build.score, 2)}"    if args.debug:
    #         print('Processed Datasets')
    #         print(fs.output.processed_datasets))

    # Probabilities
    # Calculate the cumulative probability that a hit remains in the site using the event score quantile table
    hit_in_site_probabilities = get_hit_in_site_probabilities(pandda_events, ranking, sites, event_score_quantiles)

    # Output the event and site tables
    output_tables(datasets, pandda_events, ranking, sites, hit_in_site_probabilities, fs, existing_events, existing_sites)
    time_pandda_finish = time.time()
    # TODO: Log properly
    print(f"PanDDA ran in: {round(time_pandda_finish - time_pandda_begin, 2)} seconds!")
