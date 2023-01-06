from pandda_gemmi.analyse_interface import *
from pandda_gemmi.tables import (
    GetEventTable,
    GetSiteTable,
    SaveEvents
)

def get_event_table(console, pandda_fs_model, all_events, event_ranking, sites):
    console.start_run_summary()
    # Save the events to json
    SaveEvents()(
        all_events,
        sites,
        pandda_fs_model.events_json_file
    )
    # Output a csv of the events
    # with STDOUTManager('Building and outputting event table...', f'\tDone!'):
    # event_table: EventTableInterface = EventTable.from_events(all_events_sites)
    console.start_event_table_output()
    event_table: EventTableInterface = GetEventTable()(
        all_events,
        sites,
        event_ranking,
    )
    event_table.save(pandda_fs_model.analyses.pandda_analyse_events_file)
    console.summarise_event_table_output(pandda_fs_model.analyses.pandda_analyse_events_file)