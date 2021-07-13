from __future__ import annotations

from typing import *
from functools import partial
import json

import numpy as np
import multiprocessing as mp
import joblib

from pandda_gemmi.pandda_types import *


def rank_events_size(events: Dict[EventID, Event], grid: Grid):
    # Simply sort the dtags by event cluster size
    ranked_event_ids = sorted(
        events,
        key=lambda event_id: events[event_id].cluster.size(grid),
    )

    # Then construct a new dictionary
    events_ranked = {event_id: events[event_id] for event_id in ranked_event_ids}

    return events_ranked


def rank_events_size_delta(events: Dict[EventID, Event], datasets: Dict[Dtag, Dataset]):
    ...


def rank_events_cnn(events: Dict[EventID, Event]):
    ...


def rank_events_autobuild(events: Dict[EventID, Event], autobuild_results, datasets: Dict[Dtag, Dataset], pandda_fs: PanDDAFSModel):
    ...
