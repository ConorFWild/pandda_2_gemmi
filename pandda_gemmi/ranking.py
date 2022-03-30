from __future__ import annotations

from typing import *
from pandda_gemmi.analyse_interface import *

from pandda_gemmi.common import EventID, Dtag
from pandda_gemmi.dataset import Dataset
from pandda_gemmi.fs import PanDDAFSModel
from pandda_gemmi.edalignment import Grid
from pandda_gemmi.event import Event
from pandda_gemmi.autobuild import AutobuildResult


def rank_events_size(events: Dict[EventID, Event], grid: Grid):
    # Simply sort the dtags by event cluster size
    ranked_event_ids = sorted(
        events,
        key=lambda event_id: events[event_id].cluster.size(grid),
        reverse=True,
    )

    # Then construct a new dictionary
    events_ranked = {event_id: events[event_id] for event_id in ranked_event_ids}

    return events_ranked


class GetEventRankingSize(GetEventRankingSizeInterface):
    tag: Literal["size"] = "size"

    def __call__(self, events: EventsInterface, grid: GridInterface) -> EventRankingInterface:
        return rank_events_size(events, grid)

def rank_events_size_delta(events: Dict[EventID, Event], datasets: Dict[Dtag, Dataset]):
    ...


def rank_events_cnn(events: Dict[EventID, Event]):
    ...


def rank_events_autobuild(
        events: Dict[EventID, Event],
        autobuild_results: Dict[EventID, AutobuildResult],
        datasets: Dict[Dtag, Dataset],
        pandda_fs: PanDDAFSModel,
):
    # Rank events with a score
    ranked_event_ids = list(
        sorted(
            [
                event_id
                for event_id
                in events.keys()
                if len(autobuild_results[event_id].scores) != 0
            ],
            key=lambda event_id: max(autobuild_results[event_id].scores.values()),
            reverse=True,
        )
    )

    # Add events missing any autobuilds
    for event_id in autobuild_results.keys():
        if len(autobuild_results[event_id].scores) == 0:
            ranked_event_ids.append(event_id)

    events_ranked = {event_id: events[event_id] for event_id in ranked_event_ids}

    return events_ranked


class GetEventRankingAutobuild(GetEventRankingAutobuildInterface):
    tag: Literal["autobuild"] = "autobuild"

    def __call__(self, events: EventsInterface, autobuild_results: AutobuildResultsInterface, datasets: DatasetsInterface, pandda_fs_model: PanDDAFSModelInterface) -> EventRankingInterface:
        return rank_events_autobuild(events, autobuild_result, datasets, pandda_fs_model)
