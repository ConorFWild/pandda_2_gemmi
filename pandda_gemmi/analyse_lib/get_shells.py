from pandda_gemmi.analyse_interface import *
from pandda_gemmi.shells import get_shells_multiple_models
from pandda_gemmi.fs import GetShellDirs
from pandda_gemmi.fs import ShellResultFile, ShellFile


def get_shells(pandda_args, console, pandda_fs_model: PanDDAFSModelInterface, datasets, comparators):
    console.start_get_shells()
    shell_files_exist = [shell_file.path.exists() for shell_file in pandda_fs_model.shell_files.values()]
    if (len(shell_files_exist) > 0) & all(shell_files_exist):
        shells = {res: shell_file.load() for res, shell_file in pandda_fs_model.shell_files.items()}

    else:
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

        for res, shell in shells.items():
            shell_file = ShellFile(pandda_fs_model.shell_dirs.shell_dirs[res].path / "shell.pickle")
            shell_result_file = ShellResultFile(pandda_fs_model.shell_dirs.shell_dirs[res].path / "shell_result.pickle")
            pandda_fs_model.shell_files[res] = shell_file
            pandda_fs_model.shell_result_files[res] = shell_result_file

    return shells
