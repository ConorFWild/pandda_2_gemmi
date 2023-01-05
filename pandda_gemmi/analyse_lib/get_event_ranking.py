from pandda_gemmi.common import update_log, Partial
from pandda_gemmi.analyse_interface import *

from pandda_gemmi import constants
from pandda_gemmi.ranking import (
    GetEventRankingAutobuild,
    GetEventRankingSize,
    GetEventRankingSizeAutobuild,
    GetEventRankingEventScore
)


def get_event_ranking(pandda_args, console, pandda_fs_model, datasets, grid, all_events, event_scores,
                      autobuild_results, pandda_log):
    console.start_ranking()

    # Rank the events to determine the order the are displated in
    if pandda_args.rank_method == "size":
        event_ranking = GetEventRankingSize()(all_events, grid)
    elif pandda_args.rank_method == "size_delta":
        raise NotImplementedError()
        # all_events_ranked = rank_events_size_delta()
    elif pandda_args.rank_method == "cnn":
        raise NotImplementedError()
        # all_events_ranked = rank_events_cnn()

    elif pandda_args.rank_method == "event_score":
        event_ranking: EventRankingInterface = GetEventRankingEventScore()(
            all_events,
            event_scores,
            datasets,
            pandda_fs_model,
        )

    elif pandda_args.rank_method == "event_score_cutoff":
        event_ranking: EventRankingInterface = GetEventRankingEventScoreCutoff()(
            all_events,
            event_scores,
            datasets,
            pandda_fs_model,
        )

    elif pandda_args.rank_method == "autobuild":
        if not pandda_args.autobuild:
            raise Exception("Cannot rank on autobuilds if autobuild is not set!")
        else:
            event_ranking: EventRankingInterface = GetEventRankingAutobuild()(
                all_events,
                autobuild_results,
                datasets,
                pandda_fs_model,
            )
    elif pandda_args.rank_method == "size-autobuild":
        if not pandda_args.autobuild:
            raise Exception("Cannot rank on autobuilds if autobuild is not set!")
        else:
            event_ranking: EventRankingInterface = GetEventRankingSizeAutobuild(0.4)(
                all_events,
                autobuild_results,
                datasets,
                pandda_fs_model,
            )
    else:
        raise Exception(f"Ranking method: {pandda_args.rank_method} is unknown!")

    update_log(pandda_log, pandda_args.out_dir / constants.PANDDA_LOG_FILE)

    return event_ranking
