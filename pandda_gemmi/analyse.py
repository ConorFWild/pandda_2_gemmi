if __name__ == '__main__':


    import os
    from typing import Dict
    import time
    import psutil
    import pickle
    from shlex import split
    from pprint import PrettyPrinter
    from pathlib import Path

    import numpy as np

    import joblib

    from pandda_gemmi.config import Config
    from  pandda_gemmi import logs
    from pandda_gemmi.pandda_types import (PanDDAFSModel, Datasets, Reference, 
                                           Grid, Alignments, Shells, Xmaps, Xmap,
                                           XmapArray, Model, Dtag, Zmaps, Clusterings,
                                           Events, SiteTableFile, EventTableFile,
                                           )


    def main():
        ###################################################################
        # # Configuration
        ###################################################################
        
        args_list = split("--data_dirs=\"/dls/science/groups/i04-1/conor_dev/baz2b_test/data\" --pdb_regex=\"*.dimple.pdb\" --mtz_regex=\"*.dimple.mtz\" --out_dir=\"/dls/science/groups/i04-1/conor_dev/experiments/pandda_gemmi_test\" --structure_factors=\"2FOFCWT,PH2FOFCWT\"")

        config: Config = Config.from_args_list(args_list)
        pandda_log: logs.LogData = logs.LogData.initialise()
        pandda_log.config = config

        pandda_fs_model: PanDDAFSModel = PanDDAFSModel.from_dir(config.input.data_dirs,
                                                                config.output.out_dir,
                                                                config.input.pdb_regex,
                                                                config.input.mtz_regex,
                                                                )
        pandda_fs_model.build()
        pandda_log.fs_log = logs.FSLog.from_pandda_fs_model(pandda_fs_model)
        
        ###################################################################
        # # Pre-pandda
        ###################################################################
        
        # Get datasets
        datasets_initial: Datasets = Datasets.from_dir(pandda_fs_model)
        pandda_log.preprocessing_log.initial_datasets_log = logs.InitialDatasetLog.from_initial_datasets(datasets_initial)
        
        # Initial filters
        datasets_invalid: Datasets = datasets_initial.remove_invalid_structure_factor_datasets(
        config.params.diffraction_data.structure_factors)
        pandda_log.preprocessing_log.invalid_datasets_log = logs.InvalidDatasetLog.from_datasets(datasets_initial, datasets_invalid)
        
        datasets_low_res: Datasets = datasets_invalid.remove_low_resolution_datasets(
            config.params.diffraction_data.low_resolution_completeness)
        pandda_log.preprocessing_log.low_res_datasets_log = logs.InvalidDatasetLog.from_datasets(datasets_invalid, datasets_low_res)
        
        datasets_rfree: Datasets = datasets_low_res.remove_bad_rfree(config.params.filtering.max_rfree)
        pandda_log.preprocessing_log.rfree_datasets_log = logs.RFreeDatasetLog.from_datasets(datasets_low_res, datasets_rfree)

        datasets_wilson: Datasets = datasets_rfree.remove_bad_wilson(config.params.filtering.max_wilson_plot_z_score)  # TODO
        pandda_log.preprocessing_log.wilson_datasets_log = logs.WilsonDatasetLog.from_datasets(datasets_rfree, datasets_wilson)

        # Select refernce
        reference: Reference = Reference.from_datasets(datasets_wilson)
        pandda_log.reference_log = logs.ReferenceLog.from_reference(reference)

        
        # Post-reference filters
        datasets_smoother: Datasets = datasets_wilson.smooth_datasets(reference, 
                                                      structure_factors=config.params.diffraction_data.structure_factors,
                                                      )  
        pandda_log.preprocessing_log.smoothing_datasets_log = logs.SmoothingDatasetLog.from_datasets(datasets_invalid)

        datasets_diss_struc: Datasets = datasets_smoother.remove_dissimilar_models(reference,
                                                            config.params.filtering.max_rmsd_to_reference,
                                                            )
        pandda_log.preprocessing_log.struc_datasets_log = logs.StrucDatasetLog.from_datasets(datasets_smoother, datasets_diss_struc)

        datasets_diss_space: Datasets = datasets_diss_struc.remove_dissimilar_space_groups(reference)
        pandda_log.preprocessing_log.space_datasets_log = logs.SpaceDatasetLog.from_datasets(datasets_diss_struc, datasets_diss_space)

        datasets = datasets_diss_space

        # Grid
        grid: Grid = Grid.from_reference(reference,
                                config.params.masks.outer_mask,
                                    config.params.masks.inner_mask_symmetry,
                                    )
        pandda_log.grid_log = logs.GridLog.from_grid(grid)
        
        alignments: Alignments = Alignments.from_datasets(reference,
                                                        datasets,
                                                        )
        pandda_log.alignments_log = logs.AlignmentsLog.from_alignments(alignments)
                
        ###################################################################
        # # Process shells
        ###################################################################
        shells = Shells.from_datasets(
            datasets, 
            config.params.resolution_binning.min_characterisation_datasets,
            config.params.resolution_binning.max_shell_datasets,
            config.params.resolution_binning.high_res_increment)
        
        all_events = {}
        for shell in shells:
            pandda_log.shells_log[shell.number] = logs.ShellLog.from_shell(shell)

            # Seperate out test and train datasets
            shell_datasets: Datasets = datasets.from_dtags(shell.all_dtags)

            shell_truncated_datasets: Datasets = shell_datasets.truncate(resolution=shell.res_min,
                                                                        structure_factors=config.params.diffraction_data.structure_factors,
                                                                        )
            
            # Assign datasets
            shell_train_datasets: Datasets = shell_truncated_datasets.from_dtags(shell.train_dtags)
            shell_test_datasets: Datasets = shell_truncated_datasets.from_dtags(shell.test_dtags)

            # Generate aligned xmaps
            xmaps = Xmaps.from_aligned_datasets_c(
                shell_truncated_datasets, 
                alignments, 
                grid,
                config.params.diffraction_data.structure_factors, 
                sample_rate=4.0,
                mapper=True,
                )
            
            # Seperate out test and train maps
            shell_train_xmaps: Xmaps = xmaps.from_dtags(shell.train_dtags)
            shell_test_xmaps: Xmaps = xmaps.from_dtags(shell.test_dtags)
            
            # Get arrays for model        
            masked_xmap_array: XmapArray = XmapArray.from_xmaps(xmaps,
                                        grid,
                                        )
            masked_train_xmap_array: XmapArray = masked_xmap_array.from_dtags(shell.train_dtags)
            print(len(masked_train_xmap_array.dtag_list))
            masked_test_xmap_array: XmapArray = masked_xmap_array.from_dtags(shell.test_dtags)
            print(len(masked_test_xmap_array.dtag_list))
            
            # Determine the parameters of the model to find outlying electron density
            mean_array: np.ndarray = Model.mean_from_xmap_array(masked_train_xmap_array,
                                              )
            
            sigma_is: Dict[Dtag, float] = Model.sigma_is_from_xmap_array(masked_xmap_array,
                                                        mean_array,
                                                       1.5,
                                                       )
            pandda_log.shells_log[shell.number].sigma_is = {dtag.dtag: sigma_i 
                                                            for dtag, sigma_i 
                                                            in sigma_is.items()}
            
            sigma_s_m: np.ndarray = Model.sigma_sms_from_xmaps(masked_train_xmap_array,
                                                        mean_array,
                                                        sigma_is,
                                                        )
            
            model: Model = Model.from_mean_is_sms(mean_array,
                                sigma_is,
                                sigma_s_m,
                                grid,
                                )
            
            # Calculate z maps
            zmaps: Zmaps = Zmaps.from_xmaps(model=model,
                                        xmaps=shell_test_xmaps,
                                        )

            # Get the clustered electron desnity outliers
            clusterings: Clusterings = Clusterings.from_Zmaps(
                zmaps, 
                reference,
                grid,
                config.params.masks.contour_level,
                multiprocess=True,
                )
            pandda_log.shells_log[shell.number].initial_clusters = logs.ClusteringsLog.from_clusters(
                clusterings, grid)
           
            # Filter out small clusters
            clusterings_large: Clusterings = clusterings.filter_size(grid,
                                                            12.0)
            pandda_log.shells_log[shell.number].initial_clusters = logs.ClusteringsLog.from_clusters(
                clusterings_large, grid)
            
            # Filter out weak clusters (low peak z score)
            clusterings_peaked: Clusterings = clusterings_large.filter_peak(grid,
                                                   config.params.blob_finding.min_blob_z_peak)
            pandda_log.shells_log[shell.number].initial_clusters = logs.ClusteringsLog.from_clusters(
                clusterings_peaked, grid)
            
            # Calculate the shell events
            events: Events = Events.from_clusters(clusterings_peaked, model, xmaps, grid, 1.732)
            pandda_log.shells_log[shell.number].events = logs.EventsLog.from_events(events, grid)

            # Save the z maps
            for dtag in zmaps:
                zmap = zmaps[dtag]
                pandda_fs_model.processed_datasets.processed_datasets[dtag].z_map_file.save(zmap)

            # Save the event maps!
            events.save_event_maps(shell_truncated_datasets,
                                   alignments,
                                   xmaps,
                                   model,
                                   pandda_fs_model,
                                   grid,
                                   config.params.diffraction_data.structure_factors,
                                   config.params.masks.outer_mask,
                                   config.params.masks.inner_mask_symmetry,
                                   multiprocess=True,
                                   )

        # Get the sites and output a csv of them
        site_table: SiteTable = SiteTable.from_events(events)
        pandda_log.sites_log = logs.SitesLog.from_sites(site_table)

        # Output a csv of the events
        event_table_file: EventTableFile = EventTableFile.from_events(events)
                        


    main()


