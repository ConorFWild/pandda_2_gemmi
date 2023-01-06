from pandda_gemmi.common import Partial
from pandda_gemmi.analyse_interface import *


def get_rescored_events(pandda_fs_model, pandda_args, console, process_local, event_rescoring_function, datasets, grid,
                        all_events, event_scores, autobuild_results):
    console.start_rescoring(pandda_args.rescore_event_method)
    event_scores = {
        event_id: event_score
        for event_id, event_score
        in zip(
            all_events,
            process_local(
                [
                    Partial(
                        event_rescoring_function).paramaterise(
                        event_id,
                        event,
                        pandda_fs_model,
                        datasets[event_id.dtag],
                        event_scores[event_id],
                        autobuild_results[event_id],
                        grid,
                        debug=pandda_args.debug,
                    )
                    for event_id, event
                    in all_events.items()
                ],
            )
        )
    }
    console.summarise_rescoring(event_scores)
    return event_scores
