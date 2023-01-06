from pandda_gemmi.analyse_interface import *
from pandda_gemmi.shells import get_shells_multiple_models
from pandda_gemmi.fs import GetShellDirs


def get_shells(pandda_args, console, pandda_fs_model, datasets, comparators):
    console.start_get_shells()
    # Partition the Analysis into shells in which all datasets are being processed at a similar resolution for the
    # sake of computational efficiency
    shells: ShellsInterface = get_shells_multiple_models(
        datasets,
        comparators,
        pandda_args.min_characterisation_datasets,
        pandda_args.max_shell_datasets,
        pandda_args.high_res_increment,
        pandda_args.only_datasets,
        debug=pandda_args.debug,
    )
    if pandda_args.debug >= Debug.PRINT_SUMMARIES:
        print('Got shells that support multiple models')
        for shell_res, shell in shells.items():
            print(f'\tShell res: {shell.res}: {shell.test_dtags[:3]}')
            for cluster_num, dtags in shell.train_dtags.items():
                print(f'\t\t{cluster_num}: {dtags[:5]}')
    pandda_fs_model.shell_dirs = GetShellDirs()(pandda_fs_model.pandda_dir, shells)
    pandda_fs_model.shell_dirs.build()
    # if pandda_args.debug >= Debug.PRINT_NUMERICS:
    #     printer.pprint(shells)
    console.summarise_get_shells(shells)
    return shells