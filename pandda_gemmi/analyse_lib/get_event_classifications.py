from pandda_gemmi.common import update_log
from pandda_gemmi.analyse_interface import *
from pandda_gemmi import constants

def get_event_classifications(pandda_args, console, pandda_log, get_event_class, all_events, autobuild_results):
    console.start_classification()
    # If autobuild results are available, use them
    if get_event_class.tag == "autobuild":
        event_classifications: EventClassificationsInterface = {
            event_id: get_event_class(
                event,
                autobuild_results[event_id],
            )
            for event_id, event
            in all_events.items()
        }
    elif get_event_class.tag == "trivial":
        event_classifications: EventClassificationsInterface = {
            event_id: get_event_class(event)
            for event_id, event
            in all_events.items()
        }
    else:
        raise Exception("No event classifier specified!")
    console.summarise_event_classifications(event_classifications)
    update_log(
        pandda_log,
        pandda_args.out_dir / constants.PANDDA_LOG_FILE,
    )
    return event_classifications