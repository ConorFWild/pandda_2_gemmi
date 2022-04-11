from __future__ import annotations

import time
import typing
from typing import *
import dataclasses

from scipy.cluster.hierarchy import fclusterdata
from sklearn.cluster import DBSCAN
from joblib.externals.loky import set_loky_pickler
from pandda_gemmi.analyse_interface import AlignmentInterface, AlignmentsInterface, CrystallographicGridInterface, DatasetsInterface, EDClusteringsInterface, EventsInterface, GridInterface, ModelInterface, PanDDAFSModelInterface, ProcessorInterface, StructureFactorsInterface, XmapsInterface

set_loky_pickler('pickle')

from pandda_gemmi.analyse_interface import *
# from pandda_gemmi.pandda_functions import save_event_map
from pandda_gemmi.python_types import *
from pandda_gemmi.common import EventIDX, EventID, SiteID, Dtag, PositionsArray, delayed
from pandda_gemmi.dataset import Reference, Dataset, StructureFactors
from pandda_gemmi.edalignment import Grid, Xmap, Alignment, Xmaps, Partitioning
from pandda_gemmi.model import Zmap, Zmaps, Model
from pandda_gemmi.sites import Sites
from pandda_gemmi.density_clustering import Cluster, Clustering, Clusterings


def save_event_map(
        path: Path,
        xmap: XmapInterface,
        model: ModelInterface,
        event: EventInterface,
        dataset: DatasetInterface,
        alignment: AlignmentInterface,
        grid: GridInterface,
        structure_factors: StructureFactorsInterface,
        mask_radius: float,
        mask_radius_symmetry: float,
        partitioning: PartitioningInterface,
        sample_rate: float,
        # native_grid,
):
    reference_xmap_grid = xmap.xmap
    reference_xmap_grid_array = np.array(reference_xmap_grid, copy=True)

    # moving_xmap_grid: gemmi.FloatGrid = dataset.reflections.reflections.transform_f_phi_to_map(structure_factors.f,
    #                                                                                          structure_factors.phi,
    #                                                                                          )

    event_map_reference_grid = gemmi.FloatGrid(*[reference_xmap_grid.nu,
                                                 reference_xmap_grid.nv,
                                                 reference_xmap_grid.nw,
                                                 ]
                                               )
    event_map_reference_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")  # xmap.xmap.spacegroup
    event_map_reference_grid.set_unit_cell(reference_xmap_grid.unit_cell)

    event_map_reference_grid_array = np.array(event_map_reference_grid,
                                              copy=False,
                                              )

    mean_array = model.mean
    event_map_reference_grid_array[:, :, :] = (reference_xmap_grid_array - (event.bdc.bdc * mean_array)) / (
            1 - event.bdc.bdc)

    event_map_grid = Xmap.from_aligned_map_c(
        event_map_reference_grid,
        dataset,
        alignment,
        grid,
        structure_factors,
        mask_radius,
        partitioning,
        mask_radius_symmetry,
        sample_rate * 2,  # TODO: remove?
    )

    # # # Get the event bounding box
    # # Find the min and max positions
    # min_array = np.array(event.native_positions[0])
    # max_array = np.array(event.native_positions[0])
    # for position in event.native_positions:
    #     position_array = np.array(position)
    #     min_array = np.min(np.vstack(min_array, position_array), axis=0)
    #     max_array = np.max(np.vstack(max_array, position_array), axis=0)
    #
    #
    # # Get them as fractional bounding box
    # print(min_array)
    # print(max_array)
    # print(event.native_positions[0])
    # print(event.native_centroid)
    # print(event.cluster.centroid)
    #
    # box = gemmi.FractionalBox()
    # box.minimum = gemmi.Fractional(min_array[0], min_array[1], min_array[2])
    # box.maximum = gemmi.Fractional(max_array[0], max_array[1], max_array[2])

    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = event_map_grid.xmap
    ccp4.update_ccp4_header(2, True)
    ccp4.setup()
    # ccp4.set_extent(box)
    # ccp4.grid.symmetrize_max()
    ccp4.write_ccp4_map(str(path))


def get_event_mask_indicies(zmap: ZmapInterface, cluster_positions_array: NDArrayInterface) -> NDArrayInterface:
    # cluster_positions_array = extrema_cart_coords_array[cluster_indicies]
    positions = PositionsArray(cluster_positions_array).to_positions()
    event_mask = gemmi.Int8Grid(*zmap.shape())
    event_mask.spacegroup = zmap.spacegroup()
    event_mask.set_unit_cell(zmap.unit_cell())
    for position in positions:
        event_mask.set_points_around(position,
                                     radius=2.0,
                                     value=1,
                                     )

    # event_mask.symmetrize_max()

    event_mask_array = np.array(event_mask, copy=True, dtype=np.int8)
    event_mask_indicies = np.nonzero(event_mask_array)
    return event_mask_indicies


@dataclasses.dataclass()
class BDC:
    bdc: float
    mean_fraction: float
    feature_fraction: float

    @staticmethod
    def from_float(bdc: float):
        pass

    @staticmethod
    def from_cluster(xmap: Xmap, model: Model, cluster: Cluster, dtag: Dtag, grid: Grid,
                     min_bdc=0.0, max_bdc=0.95, steps=100):
        xmap_array = xmap.to_array(copy=True)

        cluster_indexes = cluster.event_mask_indicies

        protein_mask = np.array(grid.partitioning.protein_mask, copy=False, dtype=np.int8)
        protein_mask_indicies = np.nonzero(protein_mask)

        xmap_masked = xmap_array[protein_mask_indicies]
        mean_masked = model.mean[protein_mask_indicies]
        cluster_array = np.full(protein_mask.shape, False)
        cluster_array[cluster_indexes] = True
        cluster_mask = cluster_array[protein_mask_indicies]

        vals = {}
        for val in np.linspace(min_bdc, max_bdc, steps):
            subtracted_map = xmap_masked - val * mean_masked
            cluster_vals = subtracted_map[cluster_mask]
            # local_correlation = stats.pearsonr(mean_masked[cluster_mask],
            #                                    cluster_vals)[0]
            # local_correlation, local_offset = np.polyfit(x=mean_masked[cluster_mask], y=cluster_vals, deg=1)
            local_correlation = np.corrcoef(x=mean_masked[cluster_mask], y=cluster_vals)[0, 1]

            # global_correlation = stats.pearsonr(mean_masked,
            #                                     subtracted_map)[0]
            # global_correlation, global_offset = np.polyfit(x=mean_masked, y=subtracted_map, deg=1)
            global_correlation = np.corrcoef(x=mean_masked, y=subtracted_map)[0, 1]

            vals[val] = np.abs(global_correlation - local_correlation)

        mean_fraction = max(vals,
                            key=lambda x: vals[x],
                            )

        return BDC(
            mean_fraction,
            mean_fraction,
            1 - mean_fraction,
        )


@dataclasses.dataclass()
class Event(EventInterface):
    event_id: EventID
    site: SiteID
    bdc: BDCInterface
    cluster: EDClusterInterface
    native_centroid: Tuple[float, float, float]
    native_positions: List[Tuple[float, float, float]]

    @staticmethod
    def from_cluster(event_id: EventID,
                     cluster: Cluster,
                     site: SiteID,
                     bdc: BDC,
                     native_centroid: Tuple[float, float, float],
                     native_positions: List[Tuple[float, float, float]]):
        return Event(event_id=event_id,
                     site=site,
                     bdc=bdc,
                     cluster=cluster,
                     native_centroid=native_centroid,
                     native_positions=native_positions,
                     )


@dataclasses.dataclass()
class Events:
    events: EventsInterface
    # sites: Sites

    @staticmethod
    def from_clusters(
        clusterings: EDClusteringsInterface, 
        model: ModelInterface, 
        xmaps: XmapsInterface, 
        grid: GridInterface,
        alignment: AlignmentInterface, 
        cutoff: float,
        min_bdc: float, 
        max_bdc: float,
        mapper: Any = None):
        
        events: typing.Dict[EventID, Event] = {}
        sites: Sites = Sites.from_clusters(clusterings, cutoff)

        if mapper:
            jobs = {}
            for dtag in clusterings:
                clustering = clusterings[dtag]
                for event_idx in clustering:
                    event_idx = EventIDX(event_idx)
                    event_id = EventID(dtag, event_idx)

                    cluster = clustering[event_idx.event_idx]
                    xmap = xmaps[dtag]

                    site: SiteID = sites.event_to_site[event_id]

                    jobs[event_id] = delayed(Events.get_event)(xmap, cluster, dtag, site, event_id, model, grid,
                                                               min_bdc, max_bdc, )

            results = mapper(job for job in jobs.values())

            events = {event_id: event for event_id, event in zip(jobs.keys(), results)}

        else:
            for dtag in clusterings:
                clustering = clusterings[dtag]
                for event_idx in clustering:
                    event_idx = EventIDX(event_idx)
                    event_id = EventID(dtag, event_idx)

                    cluster = clustering[event_idx.event_idx]
                    xmap = xmaps[dtag]
                    bdc = BDC.from_cluster(xmap, model, cluster, dtag, grid, min_bdc, max_bdc, )

                    site: SiteID = sites.event_to_site[event_id]

                    # Get native centroid
                    native_centroid = alignment.reference_to_moving(
                        np.array(
                            (cluster.centroid[0],
                             cluster.centroid[1],
                             cluster.centroid[2],)).reshape(-1, 3)
                    )[0]

                    # Get native event mask
                    # event_positions = []
                    # for cluster_position in cluster.position_array:
                    #     position = grid.grid.point_to_position(grid.grid.get_point(x, y, z))
                    #     event_positions.append([position.x, position.y, position.z])

                    native_positions = alignment.reference_to_moving(cluster.cluster_positions_array)

                    event = Event.from_cluster(event_id,
                                               cluster,
                                               site,
                                               bdc,
                                               native_centroid,
                                               native_positions,
                                               )

                    events[event_id] = event

        return Events(events,)# sites)

    @staticmethod
    def get_event(xmap, cluster, dtag, site, event_id, model, grid, min_bdc, max_bdc, ):
        bdc = BDC.from_cluster(xmap, model, cluster, dtag, grid, min_bdc, max_bdc, )

        event = Event.from_cluster(
            event_id,
            cluster,
            site,
            bdc,
        )

        return event

    @staticmethod
    def from_all_events(event_dict: typing.Dict[EventID, Event], grid: Grid, cutoff: float):

        # Get the sites
        all_clusterings_dict = {}
        for event_id in event_dict:
            if event_id.dtag not in all_clusterings_dict:
                all_clusterings_dict[event_id.dtag] = {}

            all_clusterings_dict[event_id.dtag][event_id.event_idx.event_idx] = event_dict[event_id].cluster

        all_clusterings = {}
        for dtag in all_clusterings_dict:
            all_clusterings[dtag] = Clustering(all_clusterings_dict[dtag])

        clusterings = Clusterings(all_clusterings)

        sites: Sites = Sites.from_clusters(clusterings, cutoff)

        # Add sites to events
        events: typing.Dict[EventID, Event] = {}
        for event_id in event_dict:
            event = event_dict[event_id]

            for event_id_site, event_site in sites.event_to_site.items():
                if (event_id_site.dtag.dtag == event_id.dtag.dtag) and (
                        event_id_site.event_idx.event_idx == event_id.event_idx.event_idx):
                    site = event_site

            event.site = site

            events[event_id] = event

        return Events(events,)# sites)

    @staticmethod
    def from_sites(event_dict: typing.Dict[EventID, Event], sites):

        # Get the sites
        # all_clusterings_dict = {}
        # for event_id in event_dict:
        #     if event_id.dtag not in all_clusterings_dict:
        #         all_clusterings_dict[event_id.dtag] = {}
        #
        #     all_clusterings_dict[event_id.dtag][event_id.event_idx.event_idx] = event_dict[event_id].cluster
        #
        # all_clusterings = {}
        # for dtag in all_clusterings_dict:
        #     all_clusterings[dtag] = Clustering(all_clusterings_dict[dtag])
        #
        # clusterings = Clusterings(all_clusterings)
        #
        # sites: Sites = Sites.from_clusters(clusterings, cutoff)

        # Add sites to events
        events: typing.Dict[EventID, Event] = {}
        for event_id in event_dict:
            event = event_dict[event_id]

            for event_id_site, event_site in sites.event_to_site.items():
                if (event_id_site.dtag.dtag == event_id.dtag.dtag) and (
                        event_id_site.event_idx.event_idx == event_id.event_idx.event_idx):
                    site = event_site

            event.site = site

            events[event_id] = event

        return Events(events,)#sites)

    def __iter__(self):
        for event_id in self.events:
            yield event_id

    def __getitem__(self, item):
        return self.events[item]

    def save_event_maps(
            self,
            datasets: DatasetsInterface,
            alignments: AlignmentsInterface,
            xmaps: XmapsInterface,
            model: ModelInterface,
            pandda_fs_model: PanDDAFSModelInterface,
            grid: GridInterface,
            structure_factors: StructureFactorsInterface,
            outer_mask: float,
            inner_mask_symmetry: float ,
            sample_rate: float,
            native_grid: CrystallographicGridInterface,
            mapper: Optional[ProcessorInterface]=False,
    ):

        processed_datasets = {}
        for event_id in self:
            dtag = event_id.dtag
            event = self[event_id]
            string = f"""
            dtag: {dtag}
            event bdc: {event.bdc}
            centroid: {event.cluster.centroid}
            """
            if dtag not in processed_datasets:
                processed_datasets[dtag] = pandda_fs_model.processed_datasets[event_id.dtag]

            processed_datasets[dtag].event_map_files.add_event(event)

        if mapper:
            event_id_list = list(self.events.keys())

            # Get unique dtags
            event_dtag_list = []
            for event_id in event_id_list:
                dtag = event_id.dtag

                if len(
                        list(
                            filter(
                                lambda event_dtag: event_dtag.dtag == dtag.dtag,
                                event_dtag_list,
                            )
                        )
                ) == 0:
                    event_dtag_list.append(dtag)

            results = mapper(
                delayed(
                    Partitioning.from_structure_multiprocess)(
                    datasets[dtag].structure,
                    # grid,
                    native_grid,
                    outer_mask,
                    inner_mask_symmetry,
                )
                for dtag
                in event_dtag_list
            )

            partitioning_dict = {dtag: partitioning for dtag, partitioning in zip(event_dtag_list, results)}

            results = mapper(
                delayed(
                    save_event_map)(
                    processed_datasets[event_id.dtag].event_map_files[event_id.event_idx].path,
                    xmaps[event_id.dtag],
                    model,
                    self[event_id],
                    datasets[event_id.dtag],
                    alignments[event_id.dtag],
                    grid,
                    structure_factors,
                    outer_mask,
                    inner_mask_symmetry,
                    partitioning_dict[event_id.dtag],
                    sample_rate,
                )
                for event_id
                in event_id_list
            )


def add_sites_to_events(event_dict: EventsInterface, sites) -> EventsInterface:

        # Get the sites
        # all_clusterings_dict = {}
        # for event_id in event_dict:
        #     if event_id.dtag not in all_clusterings_dict:
        #         all_clusterings_dict[event_id.dtag] = {}
        #
        #     all_clusterings_dict[event_id.dtag][event_id.event_idx.event_idx] = event_dict[event_id].cluster
        #
        # all_clusterings = {}
        # for dtag in all_clusterings_dict:
        #     all_clusterings[dtag] = Clustering(all_clusterings_dict[dtag])
        #
        # clusterings = Clusterings(all_clusterings)
        #
        # sites: Sites = Sites.from_clusters(clusterings, cutoff)

        # Add sites to events
        events: typing.Dict[EventID, Event] = {}
        for event_id in event_dict:
            event = event_dict[event_id]

            for event_id_site, event_site in sites.event_to_site.items():
                if (event_id_site.dtag.dtag == event_id.dtag.dtag) and (
                        event_id_site.event_idx.event_idx == event_id.event_idx.event_idx):
                    site = event_site

            event.site = site

            events[event_id] = event

        # return Events(events, sites)
        return events