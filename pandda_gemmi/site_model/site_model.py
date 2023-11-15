import numpy as np
from scipy import spatial
from scipy.cluster.hierarchy import fclusterdata
import gemmi
# import networkx as nx


from ..interfaces import *
from pandda_gemmi.dataset import StructureArray
from pandda_gemmi.processor import Partial
from pandda_gemmi.alignment import Alignment

from .site import Site


# class HeirarchicalSiteModel:
#
#     def __init__(self, t=8.0):
#         self.t = t
#
#     def __call__(self,
#          datasets: Dict[str, DatasetInterface],
#          events: Dict[Tuple[str, int], EventInterface],
#          processor: ProcessorInterface,
#          structure_array_refs,
#                  ):
#
#         #
#         if len(events) == 0:
#             return {}
#
#         if len(events) == 1:
#             return {0: Site(
#                 list(events.keys())[0],
#                 np.mean(list(events.values())[0].pos_array, axis=1)
#             )}
#
#         # Get the array of centroids
#         centroid_array = np.array(
#             [
#                 np.mean(event.pos_array, axis=0)
#                 for event
#                 in events.values()
#             ]
#         )
#
#         #
#         clusters = fclusterdata(
#             centroid_array,
#             t=self.t,
#             criterion="distance",
#             method="centroid"
#         )
#
#         #
#         unique_clusters = np.unique(clusters, )
#
#         # Merge the events based on the clustering
#         event_array = np.array([event_id for event_id in events.keys()])
#         j = 0
#         sites = {}
#         for cluster_num in unique_clusters:
#             site_event_ids = event_array[clusters == cluster_num]
#             # print([event_id for event_id in site_event_ids])
#             site_positions = np.concatenate([events[(str(event_id[0]), int(event_id[1]))].pos_array for event_id in site_event_ids], axis=0)
#
#             sites[j] = Site(
#                 [(str(event_id[0]), int(event_id[1])) for event_id in site_event_ids],
#                 np.mean(site_positions, axis=0),
#             )
#             j += 1
#
#         return sites
# def _get_closest_transform(unaligned_centroid, transforms, com_ref, com_mov):
#     distances = []
#     for transform, com_ref_centroid, com_mov_centroid in zip(transforms.values(), com_ref.values(), com_mov.values()):
#         distances.append(
#             gemmi.Position(*com_mov_centroid).dist(gemmi.Position(*unaligned_centroid))
#         )
#
#     min_dist_index = np.argmin(distances)
#     return list(transforms.values())[min_dist_index], list(com_ref.values())[min_dist_index], list(com_mov.values())[min_dist_index]


class HeirarchicalSiteModel:

    def __init__(self, t=8.0):
        self.t = t

    def get_event_environment(
            self,
            query_pos_array,
            structure_array,
            ns
    ):
        indexes = ns.query_ball_point(
            query_pos_array,
            5.0,
        )

        chains = structure_array.chains[indexes]
        residues = structure_array.seq_ids[indexes]

        res_ids = {}
        for chain, res in zip(chains, residues):
            res_ids[(str(chain), str(res))] = True

        return [res_id for res_id in res_ids.keys()]


    def get_event_environments(self, datasets, events):
        # Get the environments for each event
        event_evenvironments = {}
        for dtag, dataset in datasets.items():
            st_arr = StructureArray.from_structure(dataset.structure)
            ns = spatial.KDTree(st_arr.positions)
            dtag_events = {event_id: event for event_id, event in events.items() if event_id[0] == dtag}


            for event_id, event in dtag_events.items():
                event_evenvironments[event_id] = self.get_event_environment(
                    event.pos_array,
                    st_arr,
                    ns,
                )

        return event_evenvironments

    def get_centroids(self, event_environments, reference_dataset):
        centroids = {}
        for event_id, environment in event_environments.items():
            # res_ids = set([res_id for _event_id in clique for res_id in event_environments[_event_id]])
            poss = [
                [atom.pos.x, atom.pos.y, atom.pos.z]
                for model in reference_dataset.structure.structure
                for chain in model
                for res in chain
                for atom in res
                if (chain.name, res.seqid.num) in environment
            ]
            if len(poss) == 0:
                centroid = [0.0,0.0,0.0]
            else:
                centroid = np.mean(
                    poss,
                    axis=1
                )
            centroids[event_id] = centroid

        return centroids

    def __call__(self,
                 datasets: Dict[str, DatasetInterface],
                 events: Dict[Tuple[str, int], EventInterface],
                 ref_dataset
                 ):

        # Handle edge cases
        if len(events) == 0:
            return {}

        if len(events) == 1:
            return {0: Site(
                list(events.keys())[0],
                np.mean(list(events.values())[0].pos_array, axis=1)
            )}

        # Find the residue environment of each event (chain and residue number)
        event_environments: Dict[Tuple[str, int], List[Tuple[str, str]]] = self.get_event_environments(datasets, events)

        # Find the site centroids against the reference
        centroids = self.get_centroids(
            event_environments,
            ref_dataset,
        )

        # Cluster the centroids
        event_id_array = np.array(
            [_event_id for _event_id in centroids.keys()]
        )
        centroid_array = np.array(
            [centroid for centroid in centroids.values()]
        )
        clusters = fclusterdata(
            centroid_array,
            t=self.t,
            criterion="distance",
            method="centroid"
        )

        # Construct the sites
        sites = {}
        for j, cluster in enumerate(np.unique(clusters)):
            cluster_event_id_array = event_id_array[clusters == cluster]
            sites[j] = Site(
                [(str(event_id[0]), int(event_id[1])) for event_id in cluster_event_id_array],
                centroid_array[j, :].flatten(),
            )

        return sites




# class HeirarchicalSiteModel:
#
#     def __init__(self, t=8.0):
#         self.t = t
#
#     def __call__(
#             self,
#             datasets: Dict[str, DatasetInterface],
#             events: Dict[Tuple[str, int], EventInterface],
#             processor: ProcessorInterface,
#             structure_array_refs,
#     ):
#
#         #
#         if len(events) == 0:
#             return {}
#
#         if len(events) == 1:
#             return {0: Site(
#                 [list(events.keys())[0], ],
#                 list(events.values())[0].centroid
#                 # np.mean(list(events.values())[0].pos_array, axis=1)
#             )}
#
#         # Choose a reference structure from highest res dataset
#         reference_dtag = min(datasets, key=lambda _dtag: datasets[_dtag].reflections.resolution())
#
#         # Get alignments to this reference
#         alignments: Dict[str, AlignmentInterface] = processor.process_dict(
#             {_dtag: Partial(Alignment.from_structure_arrays).paramaterise(
#                 _dtag,
#                 structure_array_refs[_dtag],
#                 structure_array_refs[reference_dtag],
#             ) for _dtag in datasets}
#         )
#
#         # Get the aligned centroids
#         aligned_centroids = []
#         for event_id, event in events.items():
#             _dtag, _event_idx = event_id
#             # unaligned_centroid = np.mean(event.pos_array, axis=0)
#             unaligned_centroid = event.centroid
#             alignment = alignments[_dtag]
#             transforms, com_ref, com_mov = alignment.get_transforms()
#             transform, associated_com_ref, closest_com_mov = _get_closest_transform(
#                 unaligned_centroid,
#                 transforms,
#                 com_ref,
#                 com_mov,
#             )
#             transformed_pos = transform.apply(gemmi.Position(*(unaligned_centroid - closest_com_mov)))
#             transformed_pos_np = np.array(
#                 [transformed_pos.x, transformed_pos.y, transformed_pos.z]) + associated_com_ref
#             aligned_centroids.append(transformed_pos_np.flatten())
#
#         # Get the array of centroids
#         centroid_array = np.array(
#             aligned_centroids
#         )
#
#         #
#         clusters = fclusterdata(
#             centroid_array,
#             t=self.t,
#             criterion="distance",
#             method="centroid"
#         )
#
#         #
#         unique_clusters = np.unique(clusters, )
#
#         # Merge the events based on the clustering
#         event_array = np.array([event_id for event_id in events.keys()])
#         j = 0
#         sites = {}
#         for cluster_num in unique_clusters:
#             site_event_ids = event_array[clusters == cluster_num]
#             # print([event_id for event_id in site_event_ids])
#             site_positions = np.concatenate(
#                 [events[(str(event_id[0]), int(event_id[1]))].pos_array for event_id in site_event_ids], axis=0)
#
#             sites[j] = Site(
#                 [(str(event_id[0]), int(event_id[1])) for event_id in site_event_ids],
#                 np.mean(site_positions, axis=0),
#             )
#             j += 1
#
#         return sites


class ResidueSiteModel:

    def get_event_environment(
            self,
            query_pos_array,
            structure_array,
            ns
    ):
        indexes = ns.query_ball_point(
            query_pos_array,
            5.0,
        )

        chains = structure_array.chains[indexes]
        residues = structure_array.seq_ids[indexes]

        res_ids = {}
        for chain, res in zip(chains, residues):
            res_ids[(str(chain), str(res))] = True

        return [res_id for res_id in res_ids.keys()]


    def get_event_environments(self, datasets, events):
        # Get the environments for each event
        event_evenvironments = {}
        for dtag, dataset in datasets.items():
            st_arr = StructureArray.from_structure(dataset.structure)
            ns = spatial.KDTree(st_arr.positions)
            dtag_events = {event_id: event for event_id, event in events.items() if event_id[0] == dtag}


            for event_id, event in dtag_events.items():
                event_evenvironments[event_id] = self.get_event_environment(
                    event.pos_array,
                    st_arr,
                    ns,
                )

        return event_evenvironments

    def get_overlap_graph(self, event_environments, min_overlap):
        g = nx.Graph()

        # Add the nodes
        for event_id in event_environments:
            g.add_node(event_id)

        # Form the site overlap matrix
        arr = np.zeros(
            (
                len(event_environments),
                len(event_environments),
            )
        )

        for j, event_1_id in enumerate(event_environments):
            environment_1 = event_environments[event_1_id]
            for k, event_2_id in enumerate(event_environments):
                environment_2 = event_environments[event_2_id]
                if event_1_id == event_2_id:
                    continue
                v = set(environment_1).intersection(set(environment_2))
                if len(v) > min_overlap:
                    arr[j, k] = 1

        # Complete the graph
        for idx, conn in np.ndenumerate(arr):
            x, y = idx

            if x == y:
                continue
            if conn:
                g.add_edge(x, y)

        return g


    def get_centroids(self, cliques, event_environments, reference ):
        centroids = {}
        for j, clique in enumerate(cliques):
            res_ids = set([res_id for _event_id in clique for res_id in event_environments[_event_id]])
            poss = [
                [atom.pos.x, atom.pos.y, atom.pos.z]
                for model in reference.structure.structure
                for chain in model
                for res in chain
                for atom in res
                if (chain.name, res.seqid.num) in res_ids
            ]
            if len(poss) == 0:
                centroid = [0.0,0.0,0.0]
            else:
                centroid = np.mean(
                    poss,
                    axis=1
                )
            centroids[j] = centroid

        return centroids

    def get_sites(self, event_environments, reference):
        # Get the overlap graph
        overlap_graph = self.get_overlap_graph(
            event_environments,
            3
        )

        # Get the disconnected Cliques
        cliques = list(nx.connected_components(overlap_graph))

        # Find the site centroids against the reference
        centroids = self.get_centroids(
            cliques,
            event_environments,
            reference,
        )

        # Construct the sites
        sites = {}
        for j, clique in enumerate(cliques):
            sites[j] = Site(
                [(str(event_id[0]), int(event_id[1])) for event_id in clique],
                centroids[j],
            )

        return sites




    def __call__(
            self,
            datasets: Dict[str, DatasetInterface],
            events: Dict[Tuple[str, int], EventInterface],
            ref_dataset
                 ):
        # Find the residue environment of each event (chain and residue number)
        event_environments: Dict[Tuple[str, int], List[Tuple[str, str]]] = self.get_event_environments(datasets, events)

        # Get the sites based on cliques of overlapping residues
        sites: Dict[int, Site] = self.get_sites(event_environments, ref_dataset)

        return sites

        ...
