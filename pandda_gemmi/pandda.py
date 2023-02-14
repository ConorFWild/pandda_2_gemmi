import dataclasses
import time

from pandda_gemmi.common import Partial

from pandda_gemmi.dataset import (
    StructureFactors,
    SmoothBFactors,
    DatasetsStatistics,
    drop_columns,
    GetDatasets,
    GetReferenceDataset,
)
from pandda_gemmi.event import (
    Event, Clusterings, Clustering, Events, get_event_mask_indicies,
    save_event_map,
)
from pandda_gemmi.density_clustering import (
    GetEDClustering, FilterEDClusteringsSize,
    FilterEDClusteringsPeak,
    MergeEDClusterings,
)
from pandda_gemmi.processing import DatasetResult, ModelResult, ShellResult

from pandda_gemmi.edalignment import Partitioning

from pandda_gemmi.analyse_interface import *


class PanDDAGetFSModel:
    def __init__(self, get_pandda_fs_model_func, get_dataset_smiles_func, pandda_console, log):

        self.get_pandda_fs_model_func = get_pandda_fs_model_func
        self.get_dataset_smiles = get_dataset_smiles_func
        self.console = pandda_console
        self.log = log

    def __call__(self, ):
        self.console.start_fs_model()
        time_fs_model_building_start = time.time()
        pandda_fs_model: PanDDAFSModelInterface = self.get_pandda_fs_model_func()
        pandda_fs_model.build(self.get_dataset_smiles)
        time_fs_model_building_finish = time.time()
        self.console.update_log("FS model building time", time_fs_model_building_finish - time_fs_model_building_start)
        self.console.summarise_fs_model(pandda_fs_model)
        return pandda_fs_model


class PanDDALoadDatasets:
    def __init__(self,
                 get_datasets_func,
                 get_dataset_statistics_func,
                 get_structure_factors_func,
                 pandda_console,
                 log
                 ):
        self.get_datasets = get_datasets_func
        self.get_dataset_statistics = get_dataset_statistics_func
        self.get_structure_factors_func = get_structure_factors_func
        self.console = pandda_console
        self.log = log

    def __call__(self, pandda_fs_model: PanDDAFSModelInterface):
        self.console.start_load_datasets()
        datasets_initial: DatasetsInterface = self.get_datasets(pandda_fs_model, )
        datasets_statistics: DatasetsStatisticsInterface = self.get_dataset_statistics(datasets_initial)
        self.console.summarise_datasets(datasets_initial, datasets_statistics)
        structure_factors = self.get_structure_factors_func(datasets_initial)
        return datasets_initial, structure_factors


class PanDDAFilterDatasets:
    def __init__(self,
                 filter_data_quality_func,
                 console,
                 log
                 ):
        self.filter_data_quality = filter_data_quality_func
        self.console = console
        self.log = log

    def __call__(self, datasets: DatasetsInterface):
        self.console.start_data_quality_filters()

        datasets_for_filtering: DatasetsInterface = {dtag: dataset for dtag, dataset in
                                                     datasets.items()}

        datasets_quality_filtered: DatasetsInterface = self.filter_data_quality(
            datasets_for_filtering,
            # structure_factors,
        )
        self.console.summarise_filtered_datasets(
            self.filter_data_quality.filtered_dtags
        )

        return datasets_quality_filtered


class PanDDAGetReference:

    def __init__(self,
                 get_reference,
                 console,
                 log
                 ):
        self.get_reference = get_reference
        self.console = console
        self.log = log

    def __call__(self, datasets_wilson):
        self.console.start_reference_selection()

        # Select refernce
        reference: ReferenceInterface = self.get_reference()(
            datasets_wilson,
        )
        self.console.pandda_log["Reference Dtag"] = str(reference.dtag)
        self.console.summarise_reference(reference)

        return reference


class PanDDAFilterReference:

    def __init__(self,
                 filter_reference_compatability,
                 console,
                 log
                 ):
        self.filter_reference_compatability = filter_reference_compatability
        self.console = console
        self.log = log
    def __call__(self, datasets_smoother, reference) -> DatasetsInterface:
        self.console.start_reference_comparability_filters()

        datasets_reference: DatasetsInterface = self.filter_reference_compatability(datasets_smoother, reference)
        # datasets: DatasetsInterface = {dtag: dataset for dtag, dataset in
        #                                datasets_reference.items()}
        self.console.summarise_filtered_datasets(
            self.filter_reference_compatability.filtered_dtags
        )

        return datasets_reference


class PanDDAPostprocessDatasets:

    def __init__(self,
                 smooth_func,
                 console,
                 log
                 ):
        self.smooth_func = smooth_func
        self.console = console
        self.log = log

    def __call__(self, datasets_quality_filtered, reference, structure_factors,):
        datasets_wilson: DatasetsInterface = drop_columns(
            datasets_quality_filtered,
            structure_factors,
        )

        self.console.start_b_factor_smoothing()
        # with STDOUTManager('Performing b-factor smoothing...', f'\tDone!'):
        start = time.time()
        datasets_smoother: DatasetsInterface = self.smooth_func(datasets_wilson, reference, structure_factors)

        finish = time.time()
        # pandda_log["Time to perform b factor smoothing"] = finish - start

        return datasets_smoother


class PanDDAGetGrid:

    def __init__(self, get_grid_func, console, log):
        self.get_grid_func = get_grid_func
        self.console = console
        self.log = log

    def __call__(self, reference):
        self.console.start_get_grid()

        # Grid
        # with STDOUTManager('Getting the analysis grid...', f'\tDone!'):
        grid: GridInterface = self.get_grid_func(reference,
                                                 # pandda_args.outer_mask,
                                                 # pandda_args.inner_mask_symmetry,
                                                 # # sample_rate=pandda_args.sample_rate,
                                                 # sample_rate=reference.dataset.reflections.get_resolution() / 0.5,
                                                 # debug=pandda_args.debug
                                                 )

        # if pandda_args.debug >= Debug.AVERAGE_MAPS:
        #     with open(pandda_fs_model.pandda_dir / "grid.pickle", "wb") as f:
        #         pickle.dump(grid, f)
        #
        #     grid.partitioning.save_maps(
        #         pandda_fs_model.pandda_dir
        #     )
        return grid


class PanDDAGetAlignments:
    def __init__(self,
                 get_alignments,
                 console,
                 ):
        self.get_alignments = get_alignments
        self.console = console

    def __call__(self, datasets, reference):
        self.console.start_alignments()

        alignments: AlignmentsInterface = self.get_alignments(
            reference,
            datasets,
        )

        return alignments


class PanDDAGetZmap:

    def __init__(self, get_zmap_func):
        self.get_zmap_func = get_zmap_func

    def __call__(self, xmap, model):
        ###################################################################
        # # Generate the statistical model of the dataset
        ###################################################################
        time_model_analysis_start = time.time()

        # Calculate z maps
        # if debug >= Debug.PRINT_SUMMARIES:
        #     print("\t\tCalculating zmaps")
        time_z_maps_start = time.time()
        # zmaps: ZmapsInterface = Zmaps.from_xmaps(
        #     model=model,
        #     xmaps={test_dtag: dataset_xmap, },
        #     model_number=model_number,
        #     debug=debug,
        # )
        zmap = self.get_zmap_func(model, xmap)

        # if debug >= Debug.PRINT_SUMMARIES:
        #     print("\t\tCalculated zmaps")

        time_z_maps_finish = time.time()
        # model_log[constants.LOG_DATASET_Z_MAPS_TIME] = time_z_maps_finish - time_z_maps_start
        # for dtag, zmap in zmaps.items():
        #     z_map_statistics = GetMapStatistics(
        #         zmap
        #     )
        #     model_log["ZMap statistics"] = {
        #         "mean": str(z_map_statistics.mean),
        #         "std": str(z_map_statistics.std),
        #         ">1.0": str(z_map_statistics.greater_1),
        #         ">2.0": str(z_map_statistics.greater_2),
        #         ">3.0": str(z_map_statistics.greater_3),
        #     }
        #     if debug >= Debug.PRINT_SUMMARIES:
        #         print(model_log["ZMap statistics"])

        # update_log(dataset_log, dataset_log_path)

        return zmap


class PanDDAGetEvents:

    def __init__(self,
                 get_ed_clustering_func,
                 processor,
                 ):
        self.get_ed_clustering_func = get_ed_clustering_func
        self.processor = processor

    def __call__(self, datasets, reference, grid, alignment, xmap, model, zmap):
        ###################################################################
        # # Cluster the outlying density
        ###################################################################
        time_cluster_start = time.time()

        # Get the clustered electron desnity outliers

        time_cluster_z_start = time.time()

        # if debug >= Debug.PRINT_SUMMARIES:
        #     print("\t\tClustering")

        # clusterings_list: List[EDClusteringInterface] = self.processor(
        #     [
        #         Partial(self.get_ed_clustering_func).paramaterise(
        #             zmap,
        #             reference=reference,
        #             grid=grid,
        #             # contour_level=contour_level,
        #             # cluster_cutoff_distance_multiplier=cluster_cutoff_distance_multiplier,
        #         )
        #         for dtag
        #         in zmaps
        #     ]
        # )
        time_cluster_z_finish = time.time()
        clustering = self.get_ed_clustering_func(zmap, reference, grid)

        # if debug >= Debug.PRINT_SUMMARIES:
        #     print("\t\tClustering finished")

        # if debug:
        #     model_log['Time to perform primary clustering of z map'] = time_cluster_z_finish - time_cluster_z_start
        #     model_log['time_event_mask'] = {}
        #     for j, clustering in enumerate(clusterings_list):
        #         model_log['time_cluster'] = clustering.time_cluster
        #         model_log['time_np'] = clustering.time_np
        #         model_log['time_event_masking'] = clustering.time_event_masking
        #         model_log['time_get_orth'] = clustering.time_get_orth
        #         model_log['time_fcluster'] = clustering.time_fcluster
        #         for cluster_num, cluster in clustering.clustering.items():
        #             model_log['time_event_mask'][int(cluster_num)] = cluster.time_event_mask

        # clusterings: EDClusteringsInterface = {dtag: clustering for dtag, clustering in zip(zmaps, clusterings_list)}

        # model_log[constants.LOG_DATASET_INITIAL_CLUSTERS_NUM] = sum(
        #     [len(clustering) for clustering in clusterings.values()])
        # update_log(dataset_log, dataset_log_path)
        cluster_sizes = {}
        # for dtag, clustering in clusterings.items():
        for cluster_num, cluster in clustering.clustering.items():
            cluster_sizes[int(cluster_num)] = {
                "size": float(cluster.size(grid)),
                "centroid": (float(cluster.centroid[0]), float(cluster.centroid[1]), float(cluster.centroid[2])),
            }
        # model_log[constants.LOG_DATASET_CLUSTER_SIZES] = {
        #     cluster_num: cluster_sizes[cluster_num]
        #     for j, cluster_num
        #     in enumerate(sorted(
        #         cluster_sizes, key=lambda _cluster_num: cluster_sizes[_cluster_num]["size"],
        #         reverse=True,
        #     ))
        #     if j < 10
        # }
        # update_log(dataset_log, dataset_log_path)

        # Filter out small clusters
        clustering_large: EDClusteringsInterface = FilterEDClusteringsSize()(clustering,
                                                                             grid,
                                                                             self.min_blob_volume,
                                                                             )
        # if debug >= Debug.PRINT_SUMMARIES:
        #     print("\t\tAfter filtering: large: {}".format(
        #         {dtag: len(cluster) for dtag, cluster in
        #          zip(clusterings_large, clusterings_large.values())}))
        # model_log[constants.LOG_DATASET_LARGE_CLUSTERS_NUM] = sum(
        #     [len(clustering) for clustering in clusterings_large.values()])
        # update_log(dataset_log, dataset_log_path)     

        # Filter out weak clusters (low peak z score)
        clustering_peaked: EDClusteringsInterface = FilterEDClusteringsPeak()(clustering_large,
                                                                              grid,
                                                                              self.min_blob_z_peak,
                                                                              )
        # if debug >= Debug.PRINT_SUMMARIES:
        #     print("\t\tAfter filtering: peak: {}".format(
        #         {dtag: len(cluster) for dtag, cluster in
        #          zip(clusterings_peaked, clusterings_peaked.values())}))
        # model_log[constants.LOG_DATASET_PEAKED_CLUSTERS_NUM] = sum(
        #     [len(clustering) for clustering in clusterings_peaked.values()])
        # update_log(dataset_log, dataset_log_path)

        # Add the event mask
        # for clustering_id, clustering in clusterings_peaked.items():
        for cluster_id, cluster in clustering_peaked.clustering.items():
            cluster.event_mask_indicies = get_event_mask_indicies(
                zmap,
                cluster.cluster_positions_array)

        # Merge the clusters
        clustering_merged: EDClusteringsInterface = MergeEDClusterings()(clustering_peaked)
        # if debug >= Debug.PRINT_SUMMARIES:
        #     print("\t\tAfter filtering: merged: {}".format(
        #         {dtag: len(_cluster) for dtag, _cluster in
        #          zip(clusterings_merged, clusterings_merged.values())}))
        # model_log[constants.LOG_DATASET_MERGED_CLUSTERS_NUM] = sum(
        #     [len(clustering) for clustering in clusterings_merged.values()])
        # update_log(dataset_log, dataset_log_path)

        # Log the clustering
        time_cluster_finish = time.time()
        # model_log[constants.LOG_DATASET_CLUSTER_TIME] = time_cluster_finish - time_cluster_start
        # update_log(dataset_log, dataset_log_path)

        # TODO: REMOVE: event blob analysis
        # blobfind_event_map_and_report_and_output(
        #     test_dtag,
        #     model_number,
        #     dataset_truncated_datasets[test_dtag],
        #     dataset_xmaps,
        #     zmaps[test_dtag],
        #     clusterings_large,
        #     model,
        #     dataset_xmaps,
        #     grid,
        #     alignments,
        #     max_site_distance_cutoff,
        #     min_bdc, max_bdc,
        #     reference,
        #     contour_level,
        #     cluster_cutoff_distance_multiplier,
        #     pandda_fs_model
        # )

        events = Events.from_clusters(
            clustering_large,
            model,
            xmap,
            grid,
            alignment,
            self.max_site_distance_cutoff,
            self.min_bdc,
            self.max_bdc,
            None,
        )

        return events


class PanDDAScoreEvents:
    def __init__(self,
                 score_events_func,
                 ):
        self.score_events_func = score_events_func

    def __call__(self,
                 dataset_processed_dataset,
                 dataset_xmap,
                 zmap,
                 events,
                 model,
                 grid,
                 dataset_alignment,
                 ):
        # if score_events_func.tag == "inbuilt":
        event_scores: EventScoringResultsInterface = self.score_events_func(
            # test_dtag,
            # model_number,
            dataset_processed_dataset,
            dataset_xmap,
            zmap,
            events,
            model,
            grid,
            dataset_alignment,
            # max_site_distance_cutoff,
            # min_bdc, max_bdc,
            # reference,
            # res, rate,
            # event_map_cut=2.0,
            # structure_output_folder=output_dir,
            # debug=debug
        )
        # elif score_events_func.tag == "autobuild":
        #     raise NotImplementedError()

        # else:
        #     raise Exception("No valid event selection score method!")

        # model_log['score'] = {}
        # model_log['noise'] = {}
        #
        # for event_id, event_scoring_result in event_scores.items():
        #     model_log['score'][int(event_id.event_idx)] = event_scoring_result.get_selected_structure_score()
        # model_log['noise'][int(event_num)] = noises[event_num]

        # event_scores, noises = event_score_autobuild(
        #     test_dtag,
        #     model_number,
        #     dataset_processed_dataset,
        #     dataset_xmap,
        #     events,
        #     model,
        #     grid,
        #     dataset_alignment,
        #     max_site_distance_cutoff,
        #     min_bdc, max_bdc,
        #     reference,
        #     structure_output_folder=output_dir,
        #     debug=debug
        # )

        # model_log['score'] = {}
        # model_log['noise'] = {}
        #
        # for event_id, event_scoring_result in event_scores.items():
        #     model_log['score'][int(event_id.event_idx)] = event_scoring_result.get_selected_structure_score()
        #     model_log[int(event_id.event_idx)] = event_scoring_result.log()
        # model_log['noise'][int(event_num)] = noises[event_num]

        return event_scores


@dataclasses.dataclass()
class PanDDAModelResult:
    model_result: ModelResultInterface
    model_processing_time: float


class PanDDAGetModelResult:
    def __init__(self):
        self.log = {}

    def __call__(self, zmap, events, event_scores):
        time_model_analysis_start = time.time()

        # model_results = {
        #     'zmap': zmaps[test_dtag],
        #     'clusterings': clusterings,
        #     'clusterings_large': clusterings_large,
        #     'clusterings_peaked': clusterings_peaked,
        #     'clusterings_merged': clusterings_merged,
        #     'events': events,
        #     'event_scores': event_scores,
        #     'log': model_log
        # }
        model_results: ModelResultInterface = ModelResult(
            zmap,
            # clusterings,
            # clusterings_large,
            # clusterings_peaked,
            # clusterings_merged,
            {event_id: event for event_id, event in events.events.items()},
            event_scores,
            model_log
        )

        time_model_analysis_finish = time.time()

        # model_log["Model analysis time"] = time_model_analysis_finish - time_model_analysis_start
        # if debug >= Debug.PRINT_SUMMARIES:
        #     print(f"\t\tModel analysis time: {time_model_analysis_finish - time_model_analysis_start}")

        # if debug >= Debug.PRINT_SUMMARIES:
        #     for event_id, event_score_result in model_results.event_scores.items():
        #         print(f"event log: {event_id.event_idx.event_idx} {event_id.dtag.dtag}")
        #         print(event_score_result.log())

        return PanDDAModelResult(
            model_results,
            time_model_analysis_finish - time_model_analysis_start,
        )


class PanDDAProcessModel:

    def __init__(self,
                 get_zmap,
                 get_events,
                 score_events,
                 get_model_result,
                 ):
        self.get_zmap = get_zmap
        self.get_events = get_events
        self.score_events = score_events
        self.get_model_result = get_model_result

    def __call__(self, dataset, reference, grid, alignment, xmap, model):
        zmap = self.get_zmap(xmap, model)

        events = self.get_events(dataset, reference, grid, alignment, model, zmap)

        event_scores = self.score_events(
            dataset,
            xmap,
            zmap,
            events,
            model,
            grid,
            alignment,
        )

        model_result = self.get_model_result(zmap, events, event_scores)

        return model_result


class PanDDAGetModelResults:

    def __init__(self,
                 analyse_model_func,
                 process_local,
                 ):
        self.analyse_model_func = analyse_model_func
        self.process_local = process_local

    def __call__(self, pandda_fs_model, dataset, reference, grid, alignment, xmap, models):
        ###################################################################
        # # Process the models...
        ###################################################################
        time_model_analysis_start = time.time()

        model_results: ModelResultsInterface = {
            model_number: model_result
            for model_number, model_result
            in zip(
                models,
                self.process_local(
                    [
                        Partial(
                            self.analyse_model_func).paramaterise(
                            dataset,
                            reference,
                            grid,
                            alignment,
                            # model_number,
                            # test_dtag=test_dtag,
                            xmap,
                            # dataset_processed_dataset=pandda_fs_model.processed_datasets.processed_datasets[test_dtag],
                            model,
                            # max_site_distance_cutoff=max_site_distance_cutoff,
                            # min_bdc=min_bdc, max_bdc=max_bdc,
                            # contour_level=contour_level,
                            # cluster_cutoff_distance_multiplier=cluster_cutoff_distance_multiplier,
                            # min_blob_volume=min_blob_volume,
                            # min_blob_z_peak=min_blob_z_peak,
                            # output_dir=pandda_fs_model.processed_datasets.processed_datasets[test_dtag].path,
                            # score_events_func=score_events_func,
                            # res=shell.res,
                            # rate=0.5,
                            # debug=debug
                        )
                        for model_number, model
                        in models.items()
                    ]
                )
            )
        }

        # dataset_log["Model logs"] = {model_number: model_result.model_log for model_number, model_result in
        #                              model_results.items()}  #

        time_model_analysis_finish = time.time()

        # dataset_log["Time to analyse all models"] = time_model_analysis_finish - time_model_analysis_start

        # if debug >= Debug.PRINT_SUMMARIES:
        #     print(f"\tTime to analyse all models: {time_model_analysis_finish - time_model_analysis_start}")
        #     for model_number, model_result in model_results.items():
        #         model_time = dataset_log["Model logs"][model_number]["Model analysis time"]
        #         print(f"\t\tModel {model_number} processed in {model_time}")

        return model_results


class PanDDASelectModel:
    def __init__(self,
                 select_model_func,
                 ):
        self.select_model_func = select_model_func

    def __call__(self, model_results: ModelResultsInterface,
                 grid: GridInterface,
                 pandda_fs_model: PanDDAFSModelInterface,
                 ):
        ###################################################################
        # # Decide which model to use...
        ###################################################################
        # if debug >= Debug.PRINT_SUMMARIES:
        #     print(f"\tSelecting model...")
        model_selection: ModelSelectionInterface = self.select_model_func(
            model_results,
            grid.partitioning.inner_mask,
            pandda_fs_model.processed_datasets.processed_datasets[test_dtag],
            # debug=debug,
        )
        selected_model: ModelInterface = models[model_selection.selected_model_id]
        # selected_model_clusterings = model_results[model_selection.selected_model_id].clusterings_merged
        # zmap = model_results[model_selection.selected_model_id].zmap
        # dataset_log['Selected model'] = int(model_selection.selected_model_id)
        # dataset_log['Model selection log'] = model_selection.log
        #
        # if debug >= Debug.PRINT_SUMMARIES:
        #     print(f'\tSelected model is: {model_selection.selected_model_id}')

        return selected_model


class PanDDAOutputMaps:
    def __call__(self, dataset, grid, alignment, selected_model):
        ###################################################################
        # # Output the z map
        ###################################################################
        time_output_zmap_start = time.time()

        native_grid = dataset.reflections.transform_f_phi_to_map(
            # structure_factors.f,
            # structure_factors.phi,
            # sample_rate=sample_rate,  # TODO: make this d_min/0.5?
            sample_rate=dataset.reflections.get_resolution() / 0.5
        )

        partitioning = Partitioning.from_structure_multiprocess(
            dataset_truncated_datasets[test_dtag].structure,
            native_grid,
            outer_mask,
            inner_mask_symmetry,
        )
        # pandda_fs_model.processed_datasets.processed_datasets[dtag].z_map_file.save_reference_frame_zmap(zmap)

        save_native_frame_zmap(
            pandda_fs_model.processed_datasets.processed_datasets[test_dtag].z_map_file.path,
            zmap,
            dataset_truncated_datasets[test_dtag],
            alignments[test_dtag],
            grid,
            structure_factors,
            outer_mask,
            inner_mask_symmetry,
            partitioning,
            sample_rate,
        )

        # TODO: Remove altogether
        if debug >= Debug.DATASET_MAPS:
            for model_number, model_result in model_results.items():
                save_reference_frame_zmap(
                    pandda_fs_model.processed_datasets.processed_datasets[
                        test_dtag].z_map_file.path.parent / f'{model_number}_ref.ccp4',
                    model_result.zmap
                )
                save_native_frame_zmap(
                    pandda_fs_model.processed_datasets.processed_datasets[
                        test_dtag].z_map_file.path.parent / f'{model_number}_native.ccp4',
                    model_result.zmap,
                    dataset_truncated_datasets[test_dtag],
                    alignments[test_dtag],
                    grid,
                    structure_factors,
                    outer_mask,
                    inner_mask_symmetry,
                    partitioning,
                    sample_rate,
                )

        # if statmaps:
        #     mean_map_file = MeanMapFile.from_zmap_file(
        #         pandda_fs_model.processed_datasets.processed_datasets[test_dtag].z_map_file)
        #     mean_map_file.save_native_frame_mean_map(
        #         selected_model,
        #         zmap,
        #         dataset_truncated_datasets[test_dtag],
        #         alignments[test_dtag],
        #         grid,
        #         structure_factors,
        #         outer_mask,
        #         inner_mask_symmetry,
        #         partitioning,
        #         sample_rate,
        #     )

        #     std_map_file = StdMapFile.from_zmap_file(pandda_fs_model.processed_datasets.processed_datasets[
        #                                                  test_dtag].z_map_file)
        #     std_map_file.save_native_frame_std_map(
        #         test_dtag,
        #         selected_model,
        #         zmap,
        #         dataset_truncated_datasets[test_dtag],
        #         alignments[test_dtag],
        #         grid,
        #         structure_factors,
        #         outer_mask,
        #         inner_mask_symmetry,
        #         partitioning,
        #         sample_rate,
        #     )
        time_output_zmap_finish = time.time()
        dataset_log['Time to output z map'] = time_output_zmap_finish - time_output_zmap_start

        ###################################################################
        # # Find the events
        ###################################################################
        time_event_start = time.time()
        # Calculate the shell events
        # events: Events = Events.from_clusters(
        #     selected_model_clusterings,
        #     selected_model,
        #     dataset_xmaps,
        #     grid,
        #     alignments[test_dtag],
        #     max_site_distance_cutoff,
        #     min_bdc, max_bdc,
        #     None,
        # )
        events = model_results[model_selection.selected_model_id].events

        time_event_finish = time.time()
        dataset_log[constants.LOG_DATASET_EVENT_TIME] = time_event_finish - time_event_start
        update_log(dataset_log, dataset_log_path)

        ###################################################################
        # # Generate event maps
        ###################################################################
        time_event_map_start = time.time()

        # Save the event maps!
        # printer.pprint(events)
        Events(events).save_event_maps(
            dataset_truncated_datasets,
            alignments,
            dataset_xmaps,
            selected_model,
            pandda_fs_model,
            grid,
            structure_factors,
            outer_mask,
            inner_mask_symmetry,
            sample_rate,
            native_grid,
            mapper=ProcessLocalSerial(),
        )

        if debug >= Debug.DATASET_MAPS:
            for model_number, model_result in model_results.items():
                for event_id, event in model_result.events.items():
                    save_event_map(
                        pandda_fs_model.processed_datasets.processed_datasets[event_id.dtag].path / f'{model_number}'
                                                                                                    f'_{event_id.event_idx.event_idx}.ccp4',
                        dataset_xmaps[event_id.dtag],
                        models[model_number],
                        event,
                        dataset_truncated_datasets[event_id.dtag],
                        alignments[event_id.dtag],
                        grid,
                        structure_factors,
                        outer_mask,
                        inner_mask_symmetry,
                        partitioning,
                        sample_rate,
                    )

        time_event_map_finish = time.time()
        dataset_log[constants.LOG_DATASET_EVENT_MAP_TIME] = time_event_map_finish - time_event_map_start
        update_log(dataset_log, dataset_log_path)


class PanDDAGetDatasetResult:
    def __call__(self, *args, **kwargs):
        time_dataset_finish = time.time()
        dataset_log[constants.LOG_DATASET_TIME] = time_dataset_finish - time_dataset_start
        update_log(dataset_log, dataset_log_path)

        return DatasetResult(
            dtag=test_dtag,
            events={event_id: event for event_id, event in events.items()},
            event_scores=model_results[model_selection.selected_model_id].event_scores,
            log=dataset_log,
        )


class PanDDAProcessDataset:
    def __init__(self,
                 process_models,
                 select_model,
                 output_maps,
                 get_dataset_result,
                 ):
        self.process_models = process_models
        self.select_model = select_model
        self.output_maps = output_maps
        self.get_dataset_result = get_dataset_result

        self.log = log

    def __call__(self,
                 dataset,
                 shell,
                 grid,
                 alignment,
                 xmap,
                 models, ):
        model_results = self.process_models(dataset, grid, alignment, xmap, models)

        selected_model = self.select_model(model_results)

        self.output_maps(dataset, grid, alignment, selected_model)

        dataset_result = self.get_dataset_result(dataset, selected_model)

        return dataset_result


class PanDDAGetShellDatasets:
    def __call__(self, *args, **kwargs):
        # Seperate out test and train datasets
        shell_datasets: DatasetsInterface = {
            dtag: dataset
            for dtag, dataset
            in datasets.items()
            if dtag in shell.all_dtags
        }
        shell_log[constants.LOG_SHELL_DATASETS] = [dtag.dtag for dtag in shell_datasets]
        update_log(shell_log, shell_log_path)

        return shell_datasets


class PanDDAGetHomogenisedDatasets:
    def __init__(self,
                 truncate_dataset_reflections_func,
                 ):
        self.truncate_dataset_reflections_func = truncate_dataset_reflections_func

    def __call__(self, datasets, shell):
        ###################################################################
        # # Homogonise shell datasets by truncation of resolution
        ###################################################################
        # if debug >= Debug.PRINT_SUMMARIES:
        #     print(f"\tTruncating shell datasets")
        shell_working_resolution: ResolutionInterface = Resolution(
            max([datasets[dtag].reflections.get_resolution() for dtag in shell.all_dtags]))
        shell_truncated_datasets: DatasetsInterface = self.truncate_dataset_reflections_func(
            shell_datasets,
            resolution=shell_working_resolution,
            structure_factors=structure_factors,
        )
        # TODO: REMOVE?
        # shell_truncated_datasets = shell_datasets
        # shell_log["Shell Working Resolution"] = shell_working_resolution.resolution

        return shell_truncated_datasets


class PanDDAGetShellXMaps:
    def __init__(self,
                 load_xmap_func,
                 process_local_in_shell,
                 ):
        self.load_xmap_func = load_xmap_func
        self.process_local_in_shell = process_local_in_shell

    def __call__(self, *args, **kwargs):
        ###################################################################
        # # Generate aligned Xmaps
        ###################################################################
        # if debug >= Debug.PRINT_SUMMARIES:
        #     print(f"\tLoading xmaps")

        time_xmaps_start = time.time()

        xmaps: XmapsInterface = {
            dtag: xmap
            for dtag, xmap
            in zip(
                shell_truncated_datasets,
                self.process_local_in_shell(
                    [
                        Partial(self.load_xmap_func).paramaterise(
                            shell_truncated_datasets[key],
                            alignments[key],
                            grid=grid,
                            # structure_factors=structure_factors,
                            # sample_rate=shell.res / 0.5,
                        )
                        for key
                        in shell_truncated_datasets
                    ]
                )
            )
        }

        time_xmaps_finish = time.time()
        # shell_log[constants.LOG_SHELL_XMAP_TIME] = time_xmaps_finish - time_xmaps_start
        # update_log(shell_log, shell_log_path)

        return xmaps


class PanDDAGetModels:
    def __init__(self,
                 get_models_func,
                 process_local_in_shell,
                 ):
        self.get_models_func = get_models_func
        self.process_local_in_shell = process_local_in_shell

    def __call__(self, shell, xmaps, grid, ):
        ###################################################################
        # # Get the models to test
        ###################################################################
        # if debug >= Debug.PRINT_SUMMARIES:
        #     print(f"\tGetting models")
        models: ModelsInterface = self.get_models_func(
            shell.test_dtags,
            shell.train_dtags,
            xmaps,
            grid,
            self.process_local_in_shell,
        )

        # if debug >= Debug.PRINT_SUMMARIES:
        #     for model_key, model in models.items():
        #         save_array_to_map_file(
        #             model.mean,
        #             grid.grid,
        #             pandda_fs_model.pandda_dir / f"{shell.res}_{model_key}_mean.ccp4"
        #         )

        return models



class PanDDAGetShellResults:
    def __init__(self,
                 get_comparators,
                 get_shells,
                 process_shell,
                 console,
                 ):

        self.get_comparators = get_comparators
        self.get_shells = get_shells
        self.process_shell = process_shell

        self.log = {}

    def __call__(self, datassets, reference, grid, alignments, ):

        shell_dataset_model_futures = {}
        model_caches = {}
        shell_truncated_datasets_cache = {}
        shell_xmaps_chace = {}

        for res, shell in shells.items():

            shell_datasets: DatasetsInterface = {
                dtag: dataset
                for dtag, dataset
                in datasets.items()
                if dtag in shell.all_dtags
            }

            shell_truncated_datasets = get_shell_datasets(
                console,
                datasets,
                structure_factors,
                shell,
                shell_datasets,
                shell_log,
                pandda_args.debug,
            )
            shell_truncated_datasets_cache[res] = cache(
                pandda_fs_model.shell_dirs.shell_dirs[res].path,
                shell_truncated_datasets,
            )

            test_datasets_to_load = {_dtag: _dataset for _dtag, _dataset in shell_truncated_datasets.items() if
                                     _dtag in shell.test_dtags}

            test_xmaps = get_xmaps(
                console,
                pandda_fs_model,
                process_local_in_shell,
                load_xmap_func,
                structure_factors,
                alignments,
                grid,
                shell,
                test_datasets_to_load,
                pandda_args.sample_rate,
                shell_log,
                pandda_args.debug,
                shell_log_path,
            )
            shell_xmaps_chace[res] = cache(
                pandda_fs_model.shell_dirs.shell_dirs[res].path,
                test_xmaps,
            )
            for model_number, comparators in shell.train_dtags.items():
                train_datasets_to_load = {_dtag: _dataset for _dtag, _dataset in shell_truncated_datasets.items() if
                                          _dtag in comparators}
                train_xmaps = get_xmaps(
                    console,
                    pandda_fs_model,
                    process_local_in_shell,
                    load_xmap_func,
                    structure_factors,
                    alignments,
                    grid,
                    shell,
                    train_datasets_to_load,
                    pandda_args.sample_rate,
                    shell_log,
                    pandda_args.debug,
                    shell_log_path,
                )

                model: ModelInterface = get_models(
                    shell.test_dtags,
                    {model_number: comparators, },
                    test_xmaps | train_xmaps,
                    grid,
                    process_local
                )[model_number]
                model_caches[(res, model_number)] = cache(
                    pandda_fs_model.shell_dirs.shell_dirs[res].path,
                    model,
                )

                for test_dtag in shell.test_dtags:
                    future_id = (res, test_dtag, model_number)
                    shell_dataset_model_futures[future_id] = process_global.submit(
                        Partial(
                            analyse_model_func).paramaterise(
                            model,
                            model_number,
                            test_dtag=test_dtag,
                            dataset=shell_truncated_datasets[test_dtag],
                            dataset_xmap=test_xmaps[test_dtag],
                            reference=reference,
                            grid=grid,
                            dataset_processed_dataset=pandda_fs_model.processed_datasets.processed_datasets[test_dtag],
                            dataset_alignment=alignments[test_dtag],
                        )
                    )

        model_results = {_future_id: future.get() for _future_id, future in shell_dataset_model_futures.items()}

        shell_results = assemble_results()

        all_events: EventsInterface = get_events()

        # Add the event maps to the fs
        for event_id, event in all_events.items():
            pandda_fs_model.processed_datasets.processed_datasets[event_id.dtag].event_map_files.add_event(event)

        return shell_results, all_events, event_scores


class PanDDAGetAutobuilds:
    def __init__(self,
                 autobuild_func,
                 console
                 ):
        self.autobuild_func = autobuild_func
        self.console = console

    def __call__(self, datasets, pandda_fs_model, events):
        autobuild_results = process_autobuilds(
            {event_id: Partial(autobuild_func).paramaterise(
                datasets[event_id.dtag],
                events[event_id],
            )
                for event_id
                in all_events
            },
        )

        return autobuild_results


class PanDDASummariseRun:

    def __init__(self,
                 get_event_class,
                 get_event_ranking,
                 get_sites,
                 save_events,
                 get_event_table,
                 get_site_table,
                 console,
                 ):
        self.get_event_class = get_event_class
        self.get_event_ranking = get_event_ranking
        self.get_sites = get_sites
        self.save_events = save_events
        self.get_event_table = get_event_table
        self.get_site_table = get_site_table
        self.console = console

    def __call__(self, *args, **kwargs):
        ###################################################################
        # # Classify Events
        ###################################################################
        self.console.start_classification()

        # If autobuild results are available, use them
        if get_event_class.tag == "autobuild":
            event_classifications: EventClassificationsInterface = {
                event_id: get_event_class(
                    event,
                    autobuild_results[event_id],
                )
                for event_id, event
                in all_events.items()
            }
        elif get_event_class.tag == "trivial":
            event_classifications: EventClassificationsInterface = {
                event_id: get_event_class(event)
                for event_id, event
                in all_events.items()
            }
        else:
            raise Exception("No event classifier specified!")

        self.console.summarise_event_classifications(event_classifications)

        # update_log(
        #     pandda_log,
        #     pandda_args.out_dir / constants.PANDDA_LOG_FILE,
        # )

        ###################################################################
        # # Rank Events
        ###################################################################
        self.console.start_ranking()

        # Rank the events to determine the order the are displated in
        with STDOUTManager('Ranking events...', f'\tDone!'):
            if pandda_args.rank_method == "size":
                event_ranking = GetEventRankingSize()(all_events, grid)
            elif pandda_args.rank_method == "size_delta":
                raise NotImplementedError()
                # all_events_ranked = rank_events_size_delta()
            elif pandda_args.rank_method == "cnn":
                raise NotImplementedError()
                # all_events_ranked = rank_events_cnn()

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

        # console.summarise_event_ranking(event_classifications)

        ###################################################################
        # # Assign Sites
        ###################################################################
        self.console.start_assign_sites()

        # Get the events and assign sites to them
        with STDOUTManager('Assigning sites to each event', f'\tDone!'):
            sites: SitesInterface = get_sites(
                all_events,
                grid,
            )
            all_events_sites: EventsInterface = add_sites_to_events(all_events, sites, )

        self.console.summarise_sites(sites)

        ###################################################################
        # # Output pandda summary information
        ###################################################################
        self.console.start_run_summary()

        # Save the events to json
        self.save_events(
            all_events,
            sites,
            pandda_fs_model.events_json_file
        )

        # Output a csv of the events
        # with STDOUTManager('Building and outputting event table...', f'\tDone!'):
        # event_table: EventTableInterface = EventTable.from_events(all_events_sites)
        event_table: EventTableInterface = GetEventTable()(
            all_events,
            sites,
            event_ranking,
        )
        event_table.save(pandda_fs_model.analyses.pandda_analyse_events_file)

        # Output site table
        with STDOUTManager('Building and outputting site table...', f'\tDone!'):
            # site_table: SiteTableInterface = SiteTable.from_events(all_events_sites,
            #                                                        pandda_args.max_site_distance_cutoff)
            site_table: SiteTableInterface = GetSiteTable()(all_events,
                                                            sites,
                                                            pandda_args.max_site_distance_cutoff)
            site_table.save(pandda_fs_model.analyses.pandda_analyse_sites_file)

        time_finish = time.time()
        self.console.pandda_log[constants.LOG_TIME] = time_finish - time_start

        # Output json log
        # with STDOUTManager('Saving json log with detailed information on run...', f'\tDone!'):
        if pandda_args.debug >= Debug.PRINT_SUMMARIES:
            printer.pprint(pandda_log)
        save_json_log(
            pandda_log,
            pandda_args.out_dir / constants.PANDDA_LOG_FILE,
        )

        print(f"PanDDA ran in: {time_finish - time_start}")


class PanDDA:

    def __init__(self,
                 get_fs_model,
                 load_datasets,
                 filter_datasets,
                 get_reference,
                 filter_reference,
                 postprocess_datasets,
                 get_grid,
                 get_alignments,
                 process_datasets,
                 autobuild,
                 rescore_events,
                 summarise_run,
                 ):
        self.get_fs_model = get_fs_model
        self.load_datasets = load_datasets
        self.filter_datasets = filter_datasets
        self.get_reference = get_reference
        self.filter_reference = filter_reference
        self.postprocess_datasets = postprocess_datasets
        self.get_grid = get_grid
        self.get_alignments = get_alignments
        self.process_datasets = process_datasets
        self.autobuild = autobuild
        self.rescore_events = rescore_events
        self.summarise_run = summarise_run

    def __call__(self, ):
        fs_model = self.get_fs_model()

        datasets = self.load_datasets(fs_model)

        datasets = self.filter_datasets(datasets)

        reference = self.get_reference(datasets)

        datasets = self.filter_reference(datasets, reference)

        datasets = self.postprocess_datasets(datasets, reference)

        grid = self.get_grid(reference)

        alignments = self.get_alignments(datasets, reference)

        events = self.process_datasets(datasets, reference, grid, alignments)

        autobuilds = self.autobuild(datasets, events, fs_model)

        rescored_events = self.rescore_events(events, autobuilds)

        self.summarise_run(processed_datasets, autobuilds)

    def __call__(self):
        ...
