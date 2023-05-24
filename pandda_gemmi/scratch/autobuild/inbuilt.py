import os

import numpy as np
import gemmi
from rdkit import Chem
from rdkit.Chem import AllChem
import joblib
import scipy
from scipy import spatial as spsp, optimize

from .. import constants
from ..interfaces import *

from ..fs import try_make
from ..dmaps import load_dmap, save_dmap
from ..dataset.structure import save_structure, load_structure, Structure
from .autobuild import AutobuildResult


def get_fragment_mol_from_dataset_smiles_path(dataset_smiles_path: Path):
    smiles_path = dataset_smiles_path

    # Get smiels string
    with open(str(smiles_path), "r") as f:
        smiles_string: str = str(f.read())

    # Load the mol
    m: Chem.Mol = Chem.MolFromSmiles(smiles_string)

    return m


def get_structures_from_mol(mol: Chem.Mol, max_conformers):
    fragment_structures = {}
    for i, conformer in enumerate(mol.GetConformers()):

        positions: np.ndarray = conformer.GetPositions()

        structure: gemmi.Structure = gemmi.Structure()
        model: gemmi.Model = gemmi.Model(f"{i}")
        chain: gemmi.Chain = gemmi.Chain(f"{i}")
        residue: gemmi.Residue = gemmi.Residue()
        residue.name = "LIG"
        residue.seqid = gemmi.SeqId(1, ' ')

        # Loop over atoms, adding them to a gemmi residue
        for j, atom in enumerate(mol.GetAtoms()):
            # Get the atomic symbol
            atom_symbol: str = atom.GetSymbol()
            gemmi_element: gemmi.Element = gemmi.Element(atom_symbol)

            # Get the position as a gemmi type
            pos: np.ndarray = positions[j, :]
            gemmi_pos: gemmi.Position = gemmi.Position(pos[0], pos[1], pos[2])

            # Get the
            gemmi_atom: gemmi.Atom = gemmi.Atom()
            gemmi_atom.name = atom_symbol
            gemmi_atom.pos = gemmi_pos
            gemmi_atom.element = gemmi_element

            # Add atom to residue
            residue.add_atom(gemmi_atom)

        chain.add_residue(residue)
        model.add_chain(chain)
        structure.add_model(model)

        fragment_structures[i] = structure

        if len(fragment_structures) > max_conformers:
            return fragment_structures

    return fragment_structures


def get_conformers(
        ligand_files: LigandFilesInterface,
        pruning_threshold=1.5,
        num_pose_samples=1000,
        max_conformers=10,
):
    # Decide how to load

    if ligand_files.ligand_smiles is not None:
        mol = get_fragment_mol_from_dataset_smiles_path(ligand_files.ligand_smiles)

        # Generate conformers
        mol: Chem.Mol = Chem.AddHs(mol)

        # Generate conformers
        cids = AllChem.EmbedMultipleConfs(
            mol,
            numConfs=num_pose_samples,
            pruneRmsThresh=pruning_threshold)

        # Translate to structures
        fragment_structures = get_structures_from_mol(
            mol,
            max_conformers,
        )

        return fragment_structures

    elif ligand_files.ligand_pdb is not None:
        fragment_structures = {0: load_structure(ligand_files.ligand_pdb), }

        return fragment_structures


    else:
        return {}


def get_structure_mean(structure):
    xs = []
    ys = []
    zs = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    # print(atom.pos)
                    pos: gemmi.Position = atom.pos
                    xs.append(pos.x)
                    ys.append(pos.y)
                    zs.append(pos.z)

    mean_x = np.mean(np.array(xs))
    mean_y = np.mean(np.array(ys))
    mean_z = np.mean(np.array(zs))

    return mean_x, mean_y, mean_z


def center_structure(structure, point):
    mean_x, mean_y, mean_z = get_structure_mean(structure)

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    pos: gemmi.Position = atom.pos
                    new_x = pos.x - mean_x + point[0]
                    new_y = pos.y - mean_y + point[1]
                    new_z = pos.z - mean_z + point[2]
                    atom.pos = gemmi.Position(new_x, new_y, new_z)

    return structure


def get_probe_structure(structure):
    structure_clone = structure.clone()
    structure_clone.remove_hydrogens()

    j = 0
    verticies = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom_1 in residue:
                    # print(atom_1.element.name)
                    if atom_1.element.name != "H":
                        verticies[j] = atom_1
                        j = j + 1
    edges = {}
    for atom_1_index in verticies:
        for atom_2_index in verticies:
            if atom_1_index == atom_2_index:
                continue
            atom_1 = verticies[atom_1_index]
            atom_2 = verticies[atom_2_index]
            pos_1 = atom_1.pos
            pos_2 = atom_2.pos
            distance = pos_1.dist(pos_2)
            if distance < 2.0:
                virtual_atom = gemmi.Atom()
                new_pos = gemmi.Position(
                    (pos_1.x + pos_2.x) / 2,
                    (pos_1.y + pos_2.y) / 2,
                    (pos_1.z + pos_2.z) / 2,

                )
                atom_symbol: str = "C"
                virtual_atom.name = atom_symbol
                gemmi_element: gemmi.Element = gemmi.Element(atom_symbol)
                virtual_atom.element = gemmi_element
                virtual_atom.pos = new_pos

                if atom_1_index < atom_2_index:
                    edges[(atom_1_index, atom_2_index)] = virtual_atom

                else:
                    edges[(atom_2_index, atom_1_index)] = virtual_atom

    for model in structure_clone:
        for chain in model:
            for residue in chain:
                for edge_index in edges:
                    virtual_atom = edges[edge_index]
                    residue.add_atom(virtual_atom)

    # print(f"Number of real atoms: {len(verticies)}")
    # print(f"Number of virtual atoms: {len(edges)}")
    return structure_clone


def transform_structure_array(
        structure_array,
        transform_array,
        rotation_matrix,
):
    structure_mean = np.mean(structure_array, axis=0)

    demeaned_structure = structure_array - structure_mean

    rotated_structure = np.matmul(demeaned_structure, rotation_matrix)

    transformed_array = rotated_structure + structure_mean + transform_array

    return transformed_array


def get_interpolated_values_c(
        grid,
        transformed_structure_array,
        n,
):
    vals = np.zeros(n, dtype=np.float32)

    # vals_list = \
    gemmi.interpolate_pos_array(
        grid,
        transformed_structure_array.astype(np.float32),
        vals
    )
    # print(f"Vals list: {vals_list}")
    # print(f"Vals: {vals}")

    return vals


def score_fit_nonquant_array(structure_array, grid, distance, params):
    x, y, z, rx, ry, rz = params

    x_2 = distance * x
    y_2 = distance * y
    z_2 = distance * z

    rotation = spsp.transform.Rotation.from_euler(
        "xyz",
        [
            rx * 360,
            ry * 360,
            rz * 360,
        ],
        degrees=True)
    rotation_matrix: np.ndarray = rotation.as_matrix()

    transformed_structure_array = transform_structure_array(
        structure_array,
        np.array([x_2, y_2, z_2]),
        rotation_matrix
    )

    n = structure_array.shape[0]

    vals = get_interpolated_values_c(grid, transformed_structure_array, n)

    vals[vals > 3.0] = 3.0

    score = np.sum(vals)

    return float(-score)


def transform_structure(structure, translation, rotation_matrix):
    mean_x, mean_y, mean_z = get_structure_mean(structure)
    structure_copy = structure.clone()
    structure_copy = center_structure(structure_copy, [0.0, 0.0, 0.0])

    transform: gemmi.Transform = gemmi.Transform()
    transform.mat.fromlist(rotation_matrix.tolist())
    transform.vec.fromlist([0.0, 0.0, 0.0])

    for model in structure_copy:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    pos: gemmi.Position = atom.pos
                    rotated_vec = transform.apply(pos)
                    rotated_position = gemmi.Position(rotated_vec.x, rotated_vec.y, rotated_vec.z)
                    atom.pos = rotated_position

    structure_copy = center_structure(
        structure_copy,
        [
            mean_x + translation[0],
            mean_y + translation[1],
            mean_z + translation[2]
        ]
    )

    return structure_copy


def score_conformer(
        centroid_cart,
        conformer,
        zmap_grid,
        event_fit_num_trys=10,
):
    centered_structure = center_structure(
        conformer,
        centroid_cart,
    )

    # Get the probe structure
    probe_structure = get_probe_structure(centered_structure)

    # Optimise
    structure_positions = []

    for model in probe_structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element.name != "H":
                        pos = atom.pos
                        structure_positions.append([pos.x, pos.y, pos.z])

    structure_array = np.array(structure_positions, dtype=np.float32)

    scores = []
    optimised_structures = []

    for j in range(event_fit_num_trys):
        res = optimize.differential_evolution(
            lambda params: score_fit_nonquant_array(
                structure_array,
                zmap_grid,
                1.0,
                params
            ),
            [
                (-6.0, 6.0), (-6, 6.0), (-6.0, 6.0),
                (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)
            ],
            # popsize=30,
        )

        scores.append(res.fun)

        # Get optimised fit
        x, y, z, rx, ry, rz = res.x
        rotation = spsp.transform.Rotation.from_euler(
            "xyz",
            [
                rx * 360,
                ry * 360,
                rz * 360,
            ],
            degrees=True)
        rotation_matrix: np.ndarray = rotation.as_matrix().T

        optimised_structure = transform_structure(
            centered_structure,
            [x, y, z],
            rotation_matrix
        )
        optimised_structures.append(optimised_structure)

    best_score_index = np.argmin(scores)

    best_score_fit_score = scores[best_score_index]
    best_optimised_structure = optimised_structures[best_score_index]

    return best_optimised_structure, float(best_score_fit_score)


def get_score_grid(dmap, st):
    # Get a mask of the protein
    inner_mask_grid = gemmi.Int8Grid(dmap.nu, dmap.nv, dmap.nw)
    inner_mask_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    inner_mask_grid.set_unit_cell(dmap.unit_cell)

    for atom in st.protein_atoms():
        pos = atom.pos
        inner_mask_grid.set_points_around(pos,
                                          radius=1.5,
                                          value=1,
                                          )
    inner_mask_grid_array = np.array(inner_mask_grid, copy=False)

    # Zero out density overlapping the protein
    dmap_array = np.array(dmap, copy=False)
    structure_mask_indicies = np.nonzero(inner_mask_grid_array)
    dmap_array[structure_mask_indicies] = 0.0

    return dmap


class AutobuildInbuilt:

    def __init__(self, cut=2.0):
        self.cut = cut

    def __call__(
            self,
            event: EventInterface,
            dataset: DatasetInterface,
            dmap_path,
            mtz_path,
            model_path,
            ligand_files,
            out_dir,
    ):

        # Get the structure
        st = Structure.from_path(model_path)

        # Get the scoring grid
        dmap = load_dmap(dmap_path)
        score_grid = get_score_grid(dmap, st)

        # ligand_scoring_results = {}
        # for ligand_key, ligand_files in dataset.ligand_files.items():

        # Generate conformers to score
        print(f"\tGetting conformermers!")
        conformers = get_conformers(ligand_files)
        print(f"\t\tGot {len(conformers)} conformers!")

        if len(conformers) == 0:
            return

        # Score conformers against the grid
        conformer_scores = {}
        for conformer_id, conformer in conformers.items():
            optimized_structure, score = score_conformer(
                np.mean(event.pos_array, axis=0),
                conformer,
                score_grid,
            )
            conformer_scores[conformer_id] = [optimized_structure, score]
            # print(f"\tLigand: {ligand_key}: Conformer: {conformer_id}: Score: {score}")

        # ligand_scoring_results[ligand_key] = conformer_scores

        if len(conformer_scores) == 0:
            return AutobuildResult(
                {},
                dmap_path,
                mtz_path,
                model_path,
                ligand_files.ligand_cif,
                out_dir
            )

        # Choose the best ligand
        # if len(ligand_scoring_results) == 0:
        #     return AutobuildResult(
        #         {},
        #         dmap_path,
        #         mtz_path,
        #         model_path,
        #         cif_path,
        #         out_dir
        #     )

        # best_ligand_key = max(
        #     ligand_scoring_results,
        #     key=lambda _ligand_key: max(
        #         ligand_scoring_results[_ligand_key],
        #         key=lambda _conformer_id: ligand_scoring_results[_ligand_key][_conformer_id][1],
        #     )
        # )
        #
        # best_ligand_conformer_scores = ligand_scoring_results[best_ligand_key]

        # Save the fit conformers
        # for conformer_id, (optimized_structure, score) in best_ligand_conformer_scores.items():
        #     save_structure(
        #         Structure(None, optimized_structure),
        #         out_dir / f"{conformer_id}.pdb",
        #     )
        #
        # log_result_dict = {
        #     str(out_dir / f"{conformer_id}.pdb"): score
        #     for conformer_id, (optimized_structure, score)
        #     in best_ligand_conformer_scores.items()
        # }
        for conformer_id, (optimized_structure, score) in conformer_scores.items():
            save_structure(
                Structure(None, optimized_structure),
                out_dir / f"{conformer_id}.pdb",
            )

        log_result_dict = {
            str(out_dir / f"{conformer_id}.pdb"): score
            for conformer_id, (optimized_structure, score)
            in conformer_scores.items()
        }

        # Return results
        return AutobuildResult(
            log_result_dict,
            dmap_path,
            mtz_path,
            model_path,
            ligand_files.ligand_cif,
            out_dir
        )
