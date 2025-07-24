from pandda_gemmi import constants

from .site_table import SiteTable, InspectSiteTable
from .event_table import EventTable, InspectEventTable


def output_tables(datasets, pandda_events, ranking, sites, hit_in_site_probabilities, fs, existing_events,
                  existing_sites):
    site_table = SiteTable.from_sites(sites)
    site_table.save(fs.output.analyses_dir / constants.PANDDA_ANALYSE_SITES_FILE)

    event_table = EventTable.from_events(datasets, pandda_events, ranking, sites, hit_in_site_probabilities)
    event_table.save(fs.output.analyses_dir / constants.PANDDA_ANALYSE_EVENTS_FILE)

    if existing_events is not None:
        site_table = InspectSiteTable.from_sites(sites)
        site_table.save(fs.output.analyses_dir / constants.PANDDA_INSPECT_SITES_PATH)

        for _event_id, _row in existing_events.items():
            if _event_id in pandda_events:
                pandda_events[_event_id].interesting = _row['Interesting']
                pandda_events[_event_id].ligand_placed = _row['Ligand Placed']
                pandda_events[_event_id].ligand_confidence = _row['Ligand Confidence']
                pandda_events[_event_id].comment = _row['Comment']
                pandda_events[_event_id].viewed = _row['Interesting']

        event_table = InspectEventTable.from_events(datasets, pandda_events, ranking, sites, hit_in_site_probabilities)
        event_table.save(fs.output.analyses_dir / constants.PANDDA_INSPECT_EVENTS_PATH)
