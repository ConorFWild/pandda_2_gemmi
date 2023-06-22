import os

import gemmi
import numpy as np

from pandda_gemmi import constants
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

def get_predicted_density(st, template_xmap, res):
    st.spacegroup_hm = gemmi.find_spacegroup_by_name("P 1").hm
    st.cell = template_xmap.unit_cell
    dencalc = gemmi.DensityCalculatorX()
    dencalc.d_min = 0.5
    dencalc.rate = 1
    dencalc.set_grid_cell_and_spacegroup(st)
    dencalc.put_model_density_on_grid(st[0])

    # Get the SFs at the right res
    sf = gemmi.transform_map_to_f_phi(dencalc.grid, half_l=True)
    data = sf.prepare_asu_data(dmin=res)

    # Get the grid oversampled to the right size
    approximate_structure_map = data.transform_f_phi_to_map(exact_size=[template_xmap.nu, template_xmap.nv, template_xmap.nw])

    return approximate_structure_map

def calculate_rscc(
        structure_path,
        xmap,
        res
):
    mask = gemmi.Int8Grid(xmap.nu, xmap.nv, xmap.nw)
    mask.spacegroup = gemmi.find_spacegroup_by_name("P1")
    mask.set_unit_cell(xmap.unit_cell)

    # Get the mask
    st = gemmi.read_structure(structure_path)
    for model in st:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    pos = atom.pos
                    mask.set_points_around(
                        pos,
                        radius=1.5,
                        value=1,
                    )
    mask_array = np.array(mask, copy=False)
    mask_indicies = np.nonzero(mask_array)

    # Get masked predicted ligand density
    predicted_density = get_predicted_density(st, xmap, res)
    predicted_density_array = np.array(predicted_density, copy=False)
    masked_predicted_values = predicted_density_array[mask_indicies]

    # Get masked xmap
    xmap_array = np.array(xmap, copy=False,)
    masked_xmap_values = xmap_array[mask_indicies]

    xmap_mean = np.mean(masked_xmap_values)
    predicted_mean = np.mean(masked_predicted_values)
    cov = (np.sum((masked_xmap_values-xmap_mean)*(masked_predicted_values-predicted_mean)))*(1/masked_xmap_values.size)

    rscc = cov / np.sqrt(np.var(masked_xmap_values)*np.var(masked_predicted_values))

    return rscc

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
            bdc=round(1-event.bdc, 2)
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
        print(f"{event_id[0]} : {event_id[1]} : {ligand_key} ")
        ligand_files = dataset.ligand_files[ligand_key]
        if not ligand_files.ligand_cif:
            print(f"\tSkipping!")
            continue
        ligand_cif_path = ligand_files.ligand_cif.resolve()

        if not ligand_files.ligand_pdb:
            ligand_pdb_path = None
        else:
            ligand_pdb_path = ligand_files.ligand_pdb.resolve()

        if not ligand_files.ligand_smiles:
            ligand_smiles_path = None
        else:
            ligand_smiles_path = ligand_files.ligand_smiles.resolve()

        ligand_autobuild_dir = autobuild_dir / ligand_key
        try_make(ligand_autobuild_dir)

        autobuild_result: AutobuildInterface = method(
            event,
            dataset,
            processed_dmap_path.resolve(),
            dataset.reflections.path.resolve(),
            processed_structure_path.resolve(),
            # ligand_cif_path,
            # ligand_pdb_path,
            # ligand_smiles_path,
            ligand_files,
            ligand_autobuild_dir.resolve(),
        )
        for path, score in autobuild_result.log_result_dict.items():
            rscc = calculate_rscc(
                path,
                dmap,
                dataset.reflections.resolution()
            )
            autobuild_result.log_result_dict[path] = rscc

        autobuild_results[ligand_key] = autobuild_result

    # Remove large temporaries
    os.remove(processed_structure_path)
    os.remove(processed_dmap_path)

    return autobuild_results
