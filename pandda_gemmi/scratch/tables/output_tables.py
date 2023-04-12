


from .. import constants

from .site_table import SiteTable
from .event_table import EventTable

def output_tables(pandda_events, ranking, sites, fs):

    site_table = SiteTable.from_sites(sites)
    site_table.save(fs.output.analyses_dir / constants.PANDDA_ANALYSE_SITES_FILE)

    event_table = EventTable.from_events(pandda_events, ranking, sites)
    event_table.save(fs.analyses_dir / constants.PANDDA_ANALYSE_EVENTS_FILE)