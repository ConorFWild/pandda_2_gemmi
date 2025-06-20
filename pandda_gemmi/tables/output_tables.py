from pandda_gemmi import constants

from .site_table import SiteTable
from .event_table import EventTable

def output_tables(datasets, pandda_events, ranking, sites, hit_in_site_probabilities, fs):

    site_table = SiteTable.from_sites(sites)
    site_table.save(fs.output.analyses_dir / constants.PANDDA_ANALYSE_SITES_FILE)

    event_table = EventTable.from_events(datasets, pandda_events, ranking, sites, hit_in_site_probabilities)
    event_table.save(fs.output.analyses_dir / constants.PANDDA_ANALYSE_EVENTS_FILE)