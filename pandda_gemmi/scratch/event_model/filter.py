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

        if len(events) == 1:
            return events
        # Get the centroids of each event
        event_centroid_array = np.array(
            [np.mean(event.pos_array, axis=0).flatten() for event in events.values()])

        # Cluster the event centroids
        print(f"event_centroid_array shape: {event_centroid_array.shape}")
        clusters = fclusterdata(
            event_centroid_array,
            t=self.t,
            criterion="distance",
            method="centroid"
        )

        # Get the event cluster membership array
        unique_clusters = np.unique(clusters, )

        # Merge the events based on the clustering
        event_array = np.array([event for event in events.values()])
        j = 0
        new_events = {}
        for cluster_num in unique_clusters:
            cluster_events = event_array[clusters == cluster_num]
            new_event = Event(
                np.concatenate([event.pos_array for event in cluster_events]),
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

class FilterLocallyHighestScoring:
    def __init__(self, radius=8.0):
        self.radius=radius

    def __call__(self,  events: Dict[int, EventInterface]):
        if len(events) == 0:
            return {}

        centroid_array = np.array([np.mean(event.pos_array, axis=0) for event in events.values()])

        event_id_array = np.array([event_id for event_id in events.keys()])
        print(f"Event id array shape: {event_id_array.shape}")

        masked_ids = np.zeros(event_id_array.shape, dtype=np.bool)
        j = 0
        new_events = {}
        for event_id in sorted(events, key=lambda _event_id: events[_event_id].score, reverse=True):
            if np.any(event_id_array[masked_ids] == event_id):
                # print(f"Within a radius {event_id}!")
                continue
            event = events[event_id]
            centroid = np.mean(event.pos_array, axis=0)
            distances = np.linalg.norm(centroid_array - centroid, axis=1)
            # print(f"Distances shape: {distances.shape}")

            distance_mask = distances < self.radius
            masked_ids[distance_mask] = True
            new_events[j] = event
            j = j+1
        for event_id, event in new_events.items():
            print(f"\t{event_id} : {np.mean(event.pos_array, axis=0)}")

        return new_events

