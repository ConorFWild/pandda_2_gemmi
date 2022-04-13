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

    # events_ranked = {event_id: events[event_id] for event_id in ranked_event_ids}
    events_ranked = [event_id for event_id in ranked_event_ids]

    return events_ranked


class GetEventRankingAutobuild(GetEventRankingAutobuildInterface):
    tag: Literal["autobuild"] = "autobuild"

    def __call__(self, events: EventsInterface, autobuild_results: AutobuildResultsInterface,
                 datasets: DatasetsInterface, pandda_fs_model: PanDDAFSModelInterface) -> EventRankingInterface:
        return rank_events_autobuild(events, autobuild_results, datasets, pandda_fs_model)


class GetEventRankingSizeAutobuild(GetEventRankingSizeAutobuildInterface):
    tag: Literal["size-autobuild"] = "size-autobuild"

    def __init__(self, cutoff: float):
        self.cutoff = cutoff

    def __call__(self,
                 events: EventsInterface,
                 autobuild_results: AutobuildResultsInterface,
                 datasets: DatasetsInterface,
                 pandda_fs_model: PanDDAFSModelInterface,
                 ) -> EventRankingInterface:
        # Get max score for each event
        autobuild_scores = {
            event_id: max(autobuild_results[event_id].scores.values())
            for event_id
            in events
            if len(autobuild_results[event_id].scores) > 0
        }

        # Get event sizes for each event
        event_scores = {
            event_id: event.cluster.indexes[0].size
            for event_id, event
            in events.items()
        }

        # Rank events with a score
        ranked_event_ids = list(
            sorted(
                [
                    event_id
                    for event_id
                    in autobuild_scores.keys()
                    if autobuild_scores[event_id] > self.cutoff
                ],
                key=lambda event_id: autobuild_scores[event_id],
                reverse=True,
            )
        )

        # Add remaining events by size
        for event_id in sorted(event_scores.keys(), key=lambda _event_id: event_scores[_event_id], reverse=True):
            if not len(autobuild_results[event_id].scores) > 0:  # i.e. event_id was not in autobuild_scores
                ranked_event_ids.append(event_id)

        # events_ranked = {event_id: events[event_id] for event_id in ranked_event_ids}
        events_ranked = [event_id for event_id in ranked_event_ids]

        return events_ranked

        ...
