import pickle

from pandda_gemmi.analyse_interface import *
from pandda_gemmi.edalignment import GetGrid


def get_grid(pandda_args, console, pandda_fs_model: PanDDAFSModelInterface, reference):
    console.start_get_grid()
    # Grid
    # with STDOUTManager('Getting the analysis grid...', f'\tDone!'):
    grid: GridInterface = GetGrid()(reference,
                                    pandda_args.outer_mask,
                                    pandda_args.inner_mask_symmetry,
                                    # sample_rate=pandda_args.sample_rate,
                                    sample_rate=reference.dataset.reflections.get_resolution() / 0.5,
                                    debug=pandda_args.debug
                                    )
    if pandda_args.debug >= Debug.AVERAGE_MAPS:
        with open(pandda_fs_model.pandda_dir / "grid.pickle", "wb") as f:
            pickle.dump(grid, f)

        grid.partitioning.save_maps(
            pandda_fs_model.pandda_dir
        )

    pandda_fs_model.grid_file.save(grid)

    console.summarise_get_grid(grid)
    return grid
