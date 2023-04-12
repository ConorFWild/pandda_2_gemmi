import os

from .. import constants
from ..interfaces import *

from ..fs import try_make
from ..dmaps import load_dmap, save_dmap
from ..dataset.structure import save_structure


class AutobuildResult:
    def __init__(self,
                 log_result_dict,
                 dmap_path,
                 mtz_path,
                 model_path,
                 cif_path,
                 out_dir
                 ):
        self.log_result_dict = log_result_dict
        self.dmap_path = dmap_path
        self.mtz_path = mtz_path
        self.model_path = model_path
        self.cif_path = cif_path
        self.out_dir = out_dir


def autobuild(
        event_id,
        dataset: DatasetInterface,
        event,
        preprocess_structure,
        preprocess_dmap,
        method,
        fs: PanDDAFSInterface
):
    # Setup the output directory
    autobuild_dir = fs.output.processed_datasets[event_id[0]] / f"autobuild_{event_id[1]}"
    try_make(autobuild_dir)

    # Load the relevant event map for autobuilding and preprocessing
    dmap = load_dmap(
        fs.output.processed_datasets[event_id[0]] / constants.PANDDA_EVENT_MAP_FILE.format(
            dtag=event_id[0],
            event_idx=event_id[1],
            bdc=event.bdc
        )
    )
    processed_dmap = preprocess_dmap(dmap)
    processed_dmap_path = autobuild_dir / constants.TRUNCATED_EVENT_MAP_FILE
    save_dmap(processed_dmap, processed_dmap_path)

    # Preprocess the structure
    processed_structure = preprocess_structure(dataset.structure, event)
    processed_structure_path = autobuild_dir / constants.MASKED_PDB_FILE
    save_structure(processed_structure, processed_structure_path)

    # Autobuild for each cif
    autobuild_results = {}
    for ligand_key in dataset.ligand_files:
        ligand_cif_path = dataset.ligand_files[ligand_key].ligand_cif

        ligand_autobuild_dir = autobuild_dir / ligand_key
        try_make(ligand_autobuild_dir)

        autobuild_result = method(
            dmap=processed_dmap_path,
            mtz=str(dataset.reflections.path),
            pdb=str(processed_structure_path),
            cif=str(ligand_cif_path),
            out_dir=str(ligand_autobuild_dir),
        )
        autobuild_results[ligand_key] = autobuild_result

    # Remove large temporaries
    os.remove(processed_structure_path)
    os.remove(processed_dmap_path)

    return autobuild_results
