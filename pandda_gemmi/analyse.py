if __name__ == '__main__':

    import os
    from shlex import split
    from pprint import PrettyPrinter

    from joblib import Parallel

    from pandda_gemmi.config import Config
    from pandda_gemmi.logs import Log, XmapLogs, ModelLogs
    from pandda_gemmi.pandda_types import *


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
        # datasets: Datasets = datasets.scale_reflections()  # TODO
        print("\tAfter filters (scale reflections) {} datasets".format(len(datasets.datasets)))

        reference: Reference = Reference.from_datasets(datasets)

        datasets: Datasets = datasets.remove_dissimilar_models(reference,
                                                            config.params.filtering.max_rmsd_to_reference,
                                                            )
        print("\tAfter filters (remove dissimilar models) {} datasets".format(len(datasets.datasets)))

        datasets: Datasets = datasets.remove_dissimilar_space_groups(reference)
        print("\tAfter filters (dissimilar spacegroups) {} datasets".format(len(datasets.datasets)))

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
            continue
        
        print("\tWorking on shell {}".format(shell.res_min))
        
        shell_datasets: Datasets = datasets.from_dtags(shell.all_dtags)

        print("\tTruncating datasets...")
        shell_truncated_datasets: Datasets = shell_datasets.truncate(resolution=shell.res_min,
                                                                    structure_factors=config.params.diffraction_data.structure_factors,
                                                                    )
        
        shell_smoothed_datasets: Datasets = shell_truncated_datasets.smooth(reference_reflections=shell_truncated_datasets[reference.dtag].reflections.reflections,
                                                            structure_factors=config.params.diffraction_data.structure_factors,
                                                                )

        
        shell_train_datasets: Datasets = shell_truncated_datasets.from_dtags(shell.train_dtags)
        print(len(shell_train_datasets.datasets))
        shell_test_datasets: Datasets = shell_truncated_datasets.from_dtags(shell.test_dtags)
        print(len(shell_test_datasets.datasets))
        
        print("\tGetting maps...")
        keys = list(datasets.datasets.keys())
        
        results = joblib.Parallel(n_jobs=-2, 
                                    verbose=15,
                                    backend="multiprocessing",
                                    max_nbytes=None)(
                                        joblib.delayed(Xmap.from_unaligned_dataset)(
                                            shell_smoothed_datasets[key],
                                            alignments[key],
                                            grid,
                                            config.params.diffraction_data.structure_factors,
                                            6.0,
                                            )
                                        for key
                                        in keys
                                    )
                                    
        xmaps = {keys[i]: results[i]
            for i, key
            in enumerate(keys)
            }
        xmaps = Xmaps(xmaps)
        # xmaps: Xmaps = Xmaps.from_aligned_datasets(shell_smoothed_datasets,
        #                                         alignments,
        #                                         grid,
        #                                         config.params.diffraction_data.structure_factors,
        #                                         sample_rate=6.0,
        #                                         mapper=mapper,
        #                                         )
        # print("\t\tGot {} xmaps".format(len(xmaps)))


    main()


