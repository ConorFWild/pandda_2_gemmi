from pandda_gemmi.analyse_interface import *
from pandda_gemmi.event import GetEventScoreInbuilt, add_sites_to_events, GetEventScoreSize


def get_event_sites(console, get_sites, grid, all_events):
    console.start_assign_sites()
    # Get the events and assign sites to them
    # with STDOUTManager('Assigning sites to each event', f'\tDone!'):
    sites: SitesInterface = get_sites(
        all_events,
        grid,
    )
    all_events_sites: EventsInterface = add_sites_to_events(all_events, sites, )
    console.summarise_sites(sites)
    return sites