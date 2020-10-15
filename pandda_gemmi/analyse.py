if __name__ == '__main__':


    import os
    from typing import Dict
    import time
    import psutil
    import pickle
    from shlex import split
    from pprint import PrettyPrinter

    import numpy as np

    import joblib

    from pandda_gemmi.config import Config
    from pandda_gemmi.logs import Log, XmapLogs, ModelLogs
    from pandda_gemmi.pandda_types import (PanDDAFSModel, Datasets, Reference, 
                                           Grid, Alignments, Shells, Xmaps, Xmap,
                                           XmapArray, Model, Dtag, Zmaps, Clusterings,
                                           Events, SiteTableFile, EventTableFile
    )


    def main():
        args_list = split("--data_dirs=\"/dls/science/groups/i04-1/conor_dev/baz2b_test/data\" --pdb_regex=\"*.dimple.pdb\" --mtz_regex=\"*.dimple.mtz\" --out_dir=\"/dls/science/groups/i04-1/conor_dev/experiments/pandda_gemmi_test\" --structure_factors=\"2FOFCWT,PH2FOFCWT\"")

        config: Config = Config.from_args_list(args_list)
        # log: Log = Log.from_dir(config.output.out_dir)
        # printer.pprint(config)



        pandda_fs_model: PanDDAFSModel = PanDDAFSModel.from_dir(config.input.data_dirs,
                                                                config.output.out_dir,
                                                                config.input.pdb_regex,
                                                                config.input.mtz_regex,
                                                                )


        # mapper = MapperJoblib.from_joblib()
        pandda_fs_model.build()
        
        datasets: Datasets = Datasets.from_dir(pandda_fs_model)
        print("\tGot {} datasets".format(len(datasets.datasets)))   
        
        # Initial filters
        datasets: Datasets = datasets.remove_invalid_structure_factor_datasets(
        config.params.diffraction_data.structure_factors)
        print("\tAfter filters (structure factors) {} datasets".format(len(datasets.datasets)))
        datasets: Datasets = datasets.remove_low_resolution_datasets(
            config.params.diffraction_data.low_resolution_completeness)
        print("\tAfter filters (low resolution) {} datasets".format(len(datasets.datasets)))
        datasets: Datasets = datasets.remove_bad_rfree(config.params.filtering.max_rfree)
        print("\tAfter filters (max rfree) {} datasets".format(len(datasets.datasets)))
        datasets: Datasets = datasets.remove_bad_wilson(config.params.filtering.max_wilson_plot_z_score)  # TODO
        print("\tAfter filters {} datasets".format(len(datasets.datasets)))

        # Select refernce
        reference: Reference = Reference.from_datasets(datasets)
        
        # Post-reference filters
        datasets: Datasets = datasets.smooth_datasets(reference, 
                                                      structure_factors=config.params.diffraction_data.structure_factors,
                                                      )  
        print("\tAfter filters (scale reflections) {} datasets".format(len(datasets.datasets)))
        datasets: Datasets = datasets.remove_dissimilar_models(reference,
                                                            config.params.filtering.max_rmsd_to_reference,
                                                            )
        print("\tAfter filters (remove dissimilar models) {} datasets".format(len(datasets.datasets)))
        datasets: Datasets = datasets.remove_dissimilar_space_groups(reference)
        print("\tAfter filters (dissimilar spacegroups) {} datasets".format(len(datasets.datasets)))

        # Grid
        print("Getting grid")
        grid: Grid = Grid.from_reference(reference,
                                config.params.masks.outer_mask,
                                    config.params.masks.inner_mask_symmetry,
                                    )
        print("\tGot grid")
        
        print("Getting alignments")
        alignments: Alignments = Alignments.from_datasets(reference,
                                                        datasets,
                                                        )
        print("\tGot alignments")

        for shell in Shells.from_datasets(datasets, 
                                    config.params.resolution_binning.min_characterisation_datasets,
                        config.params.resolution_binning.max_shell_datasets,
                        config.params.resolution_binning.high_res_increment):
            # Record start time
            start = time.time()        

        
            print("\tWorking on shell {}".format(shell.res_min))
            shell_datasets: Datasets = datasets.from_dtags(shell.all_dtags)

            print("\tTruncating datasets...")
            shell_truncated_datasets: Datasets = shell_datasets.truncate(resolution=shell.res_min,
                                                                        structure_factors=config.params.diffraction_data.structure_factors,
                                                                        )
            
            # shell_smoothed_datasets: Datasets = shell_truncated_datasets.smooth(reference_reflections=shell_truncated_datasets[reference.dtag].reflections.reflections,
            #                                                     structure_factors=config.params.diffraction_data.structure_factors,
            #                                                         )

            # Assign datasets
            shell_train_datasets: Datasets = shell_truncated_datasets.from_dtags(shell.train_dtags)
            print(len(shell_train_datasets.datasets))
            shell_test_datasets: Datasets = shell_truncated_datasets.from_dtags(shell.test_dtags)
            print(len(shell_test_datasets.datasets))
            
            print("\tGetting maps...")
            xmaps_dict = {}
            for dtag in shell_truncated_datasets:
                xmap = Xmap.from_unaligned_dataset_c(
                                                shell_truncated_datasets[dtag],
                                                alignments[dtag],
                                                grid,
                                                config.params.diffraction_data.structure_factors,
                                                4.0,
                                                )
                xmaps_dict[dtag] = xmap
            xmaps = Xmaps(xmaps_dict)

            shell_train_xmaps: Xmaps = xmaps.from_dtags(shell.train_dtags)
            print(len(shell_train_xmaps.xmaps))
            shell_test_xmaps: Xmaps = xmaps.from_dtags(shell.test_dtags)
            print(len(shell_test_xmaps.xmaps))
            
            # Get arrays for model        
            masked_xmap_array: XmapArray = XmapArray.from_xmaps(xmaps,
                                        grid,
                                        )
            masked_train_xmap_array: XmapArray = masked_xmap_array.from_dtags(shell.train_dtags)
            print(len(masked_train_xmap_array.dtag_list))
            masked_test_xmap_array: XmapArray = masked_xmap_array.from_dtags(shell.test_dtags)
            print(len(masked_test_xmap_array.dtag_list))
            
            mean_array: np.ndarray = Model.mean_from_xmap_array(masked_train_xmap_array,
                                              )
            
            sigma_is: Dict[Dtag, float] = Model.sigma_is_from_xmap_array(masked_xmap_array,
                                                        mean_array,
                                                       1.5,
                                                       )
            
            sigma_s_m: np.ndarray = Model.sigma_sms_from_xmaps(masked_train_xmap_array,
                                                        mean_array,
                                                        sigma_is,
                                                        )
            
            print("\tDetermining model...")
            model: Model = Model.from_mean_is_sms(mean_array,
                                sigma_is,
                                sigma_s_m,
                                                grid,
                                )
            print("\t\tGot model")
            
            print("\tGetting zmaps...")
            zmaps: Zmaps = Zmaps.from_xmaps(model=model,
                                        xmaps=shell_test_xmaps,
                                        )
            print("\t\tGot {} zmaps".format(len(zmaps)))

            print("\tGetting clusters from zmaps...")
            clusterings: Clusterings = Clusterings.from_Zmaps(zmaps, 
                                                            reference,
                                                            grid,
                                                            config.params.blob_finding.clustering_cutoff,
                                                            )
            
            print("\t\tGot {} initial clusters!".format(sum([len(clusterings[dtag]) for dtag in clusterings])))
            
            clusterings_large: Clusterings = clusterings.filter_size(grid,
                                                            12.0)
            print("\t\tGot {} big clusters!".format(sum([len(clusterings_large[dtag]) for dtag in clusterings_large])))
            
            clusterings_peaked: Clusterings = clusterings_large.filter_peak(grid,
                                                   config.params.blob_finding.min_blob_z_peak)
            print("\t\tGot {} peaked clusters!".format(sum([len(clusterings_peaked[dtag]) for dtag in clusterings_peaked])))

            for dtag in zmaps:
                print(f"\t Saving {dtag}")
                zmap = zmaps[dtag]
                pandda_fs_model.processed_datasets.processed_datasets[dtag].z_map_file.save(zmap)

            events: Events = Events.from_clusters(clusterings_peaked, model, xmaps, grid, 1.732)

            for event_id in events:
                dtag = event_id.dtag
                event = events[event_id]
                alignment = alignments[dtag]
                dataset = datasets[dtag]
            #     print(type(alignment))
                string = f"""
                dtag: {dtag}
                event bdc: {event.bdc}
                centroid: {event.cluster.centroid}
                """
                print(string)
                processed_dataset = pandda_fs_model.processed_datasets[event_id.dtag]
                processed_dataset.event_map_files.add_event(event)
                processed_dataset.event_map_files[event.event_id.event_idx].save(xmaps[dtag],
                                                                                model,
                                                                                event,
                                                                                dataset, 
                                                                                alignment,
                                                                                grid, 
                                                                                config.params.diffraction_data.structure_factors, 
                                                                                config.params.masks.outer_mask,
                                                                                config.params.masks.inner_mask_symmetry,
                                                                                
                                                                                )
                
            finish = time.time()
            print(f"Finished in {finish - start}")


        site_table_file: SiteTableFile = SiteTableFile.from_events(events)

        event_table_file: EventTableFile = EventTableFile.from_events(events)
                        
                        
            


    main()


