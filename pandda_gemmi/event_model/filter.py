import numpy as np
from scipy.cluster.hierarchy import fclusterdata
import gemmi

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
        # print(f"event_centroid_array shape: {event_centroid_array.shape}")
        clusters = fclusterdata(
            event_centroid_array,
            t=self.t,
            criterion="distance",
            # method="centroid"
            method="complete"

        )

        # Get the event cluster membership array
        unique_clusters = np.unique(clusters, )

        # Merge the events based on the clustering
        event_array = np.array([event for event in events.values()])
        j = 0
        new_events = {}
        for cluster_num in unique_clusters:
            cluster_events = event_array[clusters == cluster_num]
            pos_array = np.concatenate([event.pos_array for event in cluster_events])
            new_event = Event(
                np.concatenate(pos_array),
                np.concatenate([event.point_array for event in cluster_events]),
                np.mean(pos_array, axis=0),
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

        centroid_array = np.array(
            [
                np.mean(event.pos_array, axis=0)
                for event
                in events.values()
            ]
        )

        event_id_array = np.array([event_id for event_id in events.keys()])
        # print(f"Event id array shape: {event_id_array.shape}")

        masked_ids = np.zeros(event_id_array.shape, dtype=bool)
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
        # for event_id, event in new_events.items():
        #     print(f"\t{event_id} : {np.mean(event.pos_array, axis=0)}")

        return new_events

class FilterLocallyHighestBuildScoring:
    def __init__(self, radius=8.0):
        self.radius=radius


    def __call__(self,  events: Dict[int, EventInterface]):
        if len(events) == 0:
            return {}

        # centroid_array = np.array(
        #     [
        #         np.mean(event.pos_array, axis=0)
        #         for event
        #         in events.values()
        #     ]
        # )
        centroid_array = np.array(
            [
                event.centroid
                for event
                in events.values()
            ]
        )

        event_id_array = np.array([event_id for event_id in events.keys()])
        # print(f"Event id array shape: {event_id_array.shape}")

        masked_ids = np.zeros(event_id_array.shape, dtype=bool)
        j = 0
        new_events = {}
        for event_id in sorted(events, key=lambda _event_id: events[_event_id].score, reverse=True):
            if np.any(event_id_array[masked_ids] == event_id):
                continue
            event = events[event_id]
            centroid = np.mean(event.pos_array, axis=0)
            distances = np.linalg.norm(centroid_array - centroid, axis=1)

            distance_mask = distances < self.radius
            masked_ids[distance_mask] = True
            new_events[j] = event
            j = j+1

        return new_events


class FilterSymmetryPosBuilds:
    def __init__(self, dataset, radius=2.0, ):
        self.dataset = dataset
        self.radius =radius

    def __call__(self, events):

        new_events = {}
        for event_id, event in events.items():
            event_build, dataset = event.build, self.dataset
            st = gemmi.read_structure(str(event_build.build_path))
            ns = gemmi.NeighborSearch(st[0], dataset.reflections.reflections.cell, self.radius + 2.0).populate(include_h=False)

            dists = []
            for model in st:
                for chain in model:
                    for res in chain:
                        for atom in res:
                            atom_pos = atom.pos
                            marks = ns.find_neighbors(atom, min_dist=0.0, max_dist=self.radius+1.0)
                            print(f"\t\t\t{atom_pos.x} {atom_pos.y} {atom_pos.z}")
                            for mark in marks:
                                print(f"\t\t\t\t{mark.x} {mark.y} {mark.z} {mark.image_idx}")
                                mark_pos = gemmi.Position(mark.x, mark.y, mark.z)
                                cra = mark.to_cra(st[0])
                                original_atom_pos = cra.atom.pos
                                # Probably symmetry image, get distance to it
                                if mark_pos.dist(original_atom_pos) > 0.0001:
                                    dists.append(mark_pos.dist(atom_pos))

            print(f"Distances: {dists}")

            # Don't filter no sym atoms near
            if len(dists) == 0:
                new_events[event_id] = event
                continue

            # Filter if any sym atoms too close
            if min(dists) > self.radius:
                new_events[event_id] = event

        return new_events




class FilterLocallyHighestLargest:
    def __init__(self, radius=8.0):
        self.radius=radius

    def __call__(self,  events: Dict[int, EventInterface]):
        if len(events) == 0:
            return {}

        centroid_array = np.array(
            [
                np.mean(event.pos_array, axis=0)
                for event
                in events.values()
            ]
        )

        event_id_array = np.array([event_id for event_id in events.keys()])
        # print(f"Event id array shape: {event_id_array.shape}")

        masked_ids = np.zeros(event_id_array.shape, dtype=bool)
        j = 0
        new_events = {}
        for event_id in sorted(events, key=lambda _event_id: events[_event_id].pos_array.size, reverse=True):
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
        # for event_id, event in new_events.items():
        #     print(f"\t{event_id} : {np.mean(event.pos_array, axis=0)}")

        return new_events