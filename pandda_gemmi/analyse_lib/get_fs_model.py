import time
import pickle

from pandda_gemmi.fs import GetPanDDAFSModel
from pandda_gemmi.common import update_log
from pandda_gemmi.analyse_interface import *
from pandda_gemmi import constants


def get_fs_model(pandda_args, console, pandda_log, process_local, get_dataset_smiles, ):
    console.start_fs_model()
    time_fs_model_building_start = time.time()
    pandda_fs_model: PanDDAFSModelInterface = GetPanDDAFSModel(
        pandda_args.data_dirs,
        pandda_args.out_dir,
        pandda_args.pdb_regex,
        pandda_args.mtz_regex,
        pandda_args.ligand_dir_regex,
        pandda_args.ligand_cif_regex,
        pandda_args.ligand_pdb_regex,
        pandda_args.ligand_smiles_regex,
    )()
    pandda_fs_model.build(get_dataset_smiles, process_local=process_local)
    time_fs_model_building_finish = time.time()
    pandda_log["FS model building time"] = time_fs_model_building_finish - time_fs_model_building_start

    if pandda_args.debug >= Debug.AVERAGE_MAPS:
        with open(pandda_fs_model.pandda_dir / "pandda_fs_model.pickle", "wb") as f:
            pickle.dump(pandda_fs_model, f)

    console.summarise_fs_model(pandda_fs_model)
    update_log(pandda_log, pandda_args.out_dir / constants.PANDDA_LOG_FILE)
    if pandda_args.debug >= Debug.PRINT_NUMERICS:
        for dtag, data_dir in pandda_fs_model.data_dirs.dataset_dirs.items():
            print(dtag)
            print(data_dir.source_ligand_cif)
            print(data_dir.source_ligand_smiles)
            print(data_dir.source_ligand_pdb)

    return pandda_fs_model
