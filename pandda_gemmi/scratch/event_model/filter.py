import numpy as np
from scipy.cluster.hierarchy import fclusterdata

from .event import Event

from ..interfaces import *


class FilterSize:
    def __init__(self, reference_frame, min_size=5.0):
        self.reference_frame = reference_frame
        self.min_size = min_size

    def __call__(self, events: Dict[int, EventInterface]):
        if len(events) == 0:
            return {}
        grid = self.reference_frame.get_grid()
        volume_element = grid.unit_cell.volume / grid.point_count

        j = 0
        new_events = {}
        for event_id, event in events.items():
            event_volume = event.pos_array.shape[0] * volume_element
            print(event_volume)
            if event_volume > self.min_size:
                new_events[j] = event
                j += 1

        return new_events

    ...


class FilterCluster:
    def __init__(self, t):
        self.t = t

    def __call__(self, events: Dict[int, EventInterface]):
        if len(events) == 0:
            return {}
        # Get the centroids of each event
        event_centroid_array = np.array(
            [np.mean(event.pos_array, axis=0).flatten() for event in events.values()])
        print(event_centroid_array.shape)
        # Cluster the event centroids
        clusters = fclusterdata(
            event_centroid_array,
            t=self.t,
            criterion="distance",
            method="centroid"
        )

        # Get the event cluster membership array
        unique_clusters = np.unique(clusters, )

        # Merge the events based on the clustering
        event_array = np.array(events.values())
        j = 0
        new_events = {}
        for cluster_num in unique_clusters:
            cluster_events = event_array[clusters == cluster_num]
            new_event = Event(
                np.concat([event.pos_array for event in cluster_events]),
                0.0
            )
            new_events[j] = new_event
            j += 1

        return new_events


class FilterScore:
    def __init__(self, min_score=0.1):
        self.min_score = min_score

    def __call__(self, events: Dict[int, EventInterface]):
        if len(events) == 0:
            return {}
        j = 0
        new_events = {}
        for event_id, event in events.items():
            if event.score > self.min_score:
                new_events[j] = event
                j += 1

        return new_events
