import itertools
import os
import time

import numpy as np
import gemmi
from rdkit import Chem
from rdkit.Chem import AllChem
import joblib
import scipy
from scipy import spatial as spsp, optimize

from pandda_gemmi import constants
from ..interfaces import *

from ..fs import try_make
from ..dmaps import load_dmap, save_dmap, SparseDMap
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

bond_type_cif_to_rdkit = {
    'single': Chem.rdchem.BondType.SINGLE,
    'double': Chem.rdchem.BondType.DOUBLE,
    'triple': Chem.rdchem.BondType.TRIPLE,
    'SINGLE': Chem.rdchem.BondType.SINGLE,
    'DOUBLE': Chem.rdchem.BondType.DOUBLE,
    'TRIPLE': Chem.rdchem.BondType.TRIPLE,
    'aromatic': Chem.rdchem.BondType.AROMATIC,
    # 'deloc': Chem.rdchem.BondType.OTHER
    'deloc': Chem.rdchem.BondType.SINGLE

}

# def handle_deloc(edited_mol):
#
#     new_editable_mol = Chem.EditableMol(Chem.Mol())
#
#
#     for atom in edited_mol.GetAtoms():
#         bonds = atom.GetBonds()
#         num_deloc = 0
#         for bond in bonds:
#             if str(bond.GetBondType()) == "OTHER":
#                 num_deloc += 1
#
#         # Decide what to do with neighbourhood if delocalized center
#         if num_deloc > 1:
#
#
#
#     for atom in edited_mol.GetAtoms():
#         atom_idx = atom.GetIdx()
#         new_atom = Chem.Atom(atom.GetSymbol())
#         charge = atom.GetFormalCharge()
#         # if atom_idx in atoms_to_charge:
#         #     charge = -1
#         new_atom.SetFormalCharge(charge)
#         new_editable_mol.AddAtom(new_atom)
#
#     for bond in edited_mol.GetBonds():
#         bond_atom_1 = bond.GetBeginAtomIdx()
#         bond_atom_2 = bond.GetEndAtomIdx()
#         double_bond = False
#         for bond_idxs in bonds_to_double:
#             if (bond_atom_1 in bond_idxs) & (bond_atom_2 in bond_idxs):
#                 double_bond = True
#         if double_bond:
#             new_editable_mol.AddBond(
#                 bond_atom_1,
#                 bond_atom_2,
#                 order=bond_type_cif_to_rdkit['double']
#             )
#         else:
#             new_editable_mol.AddBond(
#                 bond_atom_1,
#                 bond_atom_2,
#                 order=bond.GetBondType()
#             )
#     new_mol = new_editable_mol.GetMol()

def get_fragment_mol_from_dataset_cif_path(dataset_cif_path: Path):
    # Open the cif document with gemmi
    cif = gemmi.cif.read(str(dataset_cif_path))

    # Create a blank rdkit mol
    mol = Chem.Mol()
    editable_mol = Chem.EditableMol(mol)

    key = "comp_LIG"
    try:
        cif['comp_LIG']
    except:
        key = "data_comp_XXX"

    # Find the relevant atoms loop
    atom_id_loop = list(cif[key].find_loop('_chem_comp_atom.atom_id'))
    atom_type_loop = list(cif[key].find_loop('_chem_comp_atom.type_symbol'))
    atom_charge_loop = list(cif[key].find_loop('_chem_comp_atom.charge'))
    if not atom_charge_loop:
        atom_charge_loop = list(cif[key].find_loop('_chem_comp_atom.partial_charge'))
        if not atom_charge_loop:
            atom_charge_loop = [0]*len(atom_id_loop)

    aromatic_atom_loop = list(cif[key].find_loop('_chem_comp_atom.aromatic'))
    if not aromatic_atom_loop:
        aromatic_atom_loop = [None]*len(atom_id_loop)

    # Get the mapping
    id_to_idx = {}
    for j, atom_id in enumerate(atom_id_loop):
        id_to_idx[atom_id] = j

    # Iteratively add the relveant atoms
    for atom_id, atom_type, atom_charge in zip(atom_id_loop, atom_type_loop, atom_charge_loop):
        if len(atom_type) > 1:
            atom_type = atom_type[0] + atom_type[1].lower()
        atom = Chem.Atom(atom_type)
        atom.SetFormalCharge(round(float(atom_charge)))
        editable_mol.AddAtom(atom)

    # Find the bonds loop
    bond_1_id_loop = list(cif[key].find_loop('_chem_comp_bond.atom_id_1'))
    bond_2_id_loop = list(cif[key].find_loop('_chem_comp_bond.atom_id_2'))
    bond_type_loop = list(cif[key].find_loop('_chem_comp_bond.type'))
    aromatic_bond_loop = list(cif[key].find_loop('_chem_comp_bond.aromatic'))
    if not aromatic_bond_loop:
        aromatic_bond_loop = [None]*len(bond_1_id_loop)

    try:
        # Iteratively add the relevant bonds
        for bond_atom_1, bond_atom_2, bond_type, aromatic in zip(bond_1_id_loop, bond_2_id_loop, bond_type_loop, aromatic_bond_loop):
            bond_type = bond_type_cif_to_rdkit[bond_type]
            if aromatic:
                if aromatic == "y":
                    bond_type = bond_type_cif_to_rdkit['aromatic']

            editable_mol.AddBond(
                id_to_idx[bond_atom_1],
                id_to_idx[bond_atom_2],
                order=bond_type
            )
    except Exception as e:
        print(e)
        print(atom_id_loop)
        print(id_to_idx)
        print(bond_1_id_loop)
        print(bond_2_id_loop)
        raise Exception

    edited_mol = editable_mol.GetMol()
    # for atom in edited_mol.GetAtoms():
    #     print(atom.GetSymbol())
    #     for bond in atom.GetBonds():
    #         print(f"\t\t{bond.GetBondType()}")
    # for bond in edited_mol.GetBonds():
    #     ba1 = bond.GetBeginAtomIdx()
    #     ba2 = bond.GetEndAtomIdx()
    #     print(f"{bond.GetBondType()} : {edited_mol.GetAtomWithIdx(ba1).GetSymbol()} : {edited_mol.GetAtomWithIdx(ba2).GetSymbol()}")  #*}")
    # print(Chem.MolToMolBlock(edited_mol))


    # HANDLE SULFONATES
    # forward_mol = Chem.ReplaceSubstructs(
    #     edited_mol,
    #     Chem.MolFromSmiles('S(O)(O)(O)'),
    #     Chem.MolFromSmiles('S(=O)(=O)(O)'),
    #     replaceAll=True,)[0]
    patt = Chem.MolFromSmarts('S(-O)(-O)(-O)')
    matches = edited_mol.GetSubstructMatches(patt)

    sulfonates = {}
    for match in matches:
        sfn = 1
        sulfonates[sfn] = {}
        on = 1
        for atom_idx in match:
            atom = edited_mol.GetAtomWithIdx(atom_idx)
            if atom.GetSymbol() == "S":
                sulfonates[sfn]["S"] = atom_idx
            else:
                atom_charge = atom.GetFormalCharge()

                if atom_charge == -1:
                    continue
                else:
                    if on == 1:
                        sulfonates[sfn]["O1"] = atom_idx
                        on += 1
                    elif on == 2:
                        sulfonates[sfn]["O2"] = atom_idx
                        on += 1
                # elif on == 3:
                #     sulfonates[sfn]["O3"] = atom_idx
    print(f"Matches to sulfonates: {matches}")

    # atoms_to_charge = [
    #     sulfonate["O3"] for sulfonate in sulfonates.values()
    # ]
    # print(f"Atom idxs to charge: {atoms_to_charge}")
    bonds_to_double =[
        (sulfonate["S"], sulfonate["O1"]) for sulfonate in sulfonates.values()
    ] + [
        (sulfonate["S"], sulfonate["O2"]) for sulfonate in sulfonates.values()
    ]
    print(f"Bonds to double: {bonds_to_double}")

    # Replace the bonds and update O3's charge
    new_editable_mol = Chem.EditableMol(Chem.Mol())
    for atom in edited_mol.GetAtoms():
        atom_idx = atom.GetIdx()
        new_atom = Chem.Atom(atom.GetSymbol())
        charge = atom.GetFormalCharge()
        # if atom_idx in atoms_to_charge:
        #     charge = -1
        new_atom.SetFormalCharge(charge)
        new_editable_mol.AddAtom(new_atom)

    for bond in edited_mol.GetBonds():
        bond_atom_1 = bond.GetBeginAtomIdx()
        bond_atom_2 = bond.GetEndAtomIdx()
        double_bond = False
        for bond_idxs in bonds_to_double:
            if (bond_atom_1 in bond_idxs) & (bond_atom_2 in bond_idxs):
                double_bond = True
        if double_bond:
            new_editable_mol.AddBond(
                bond_atom_1,
                bond_atom_2,
                order=bond_type_cif_to_rdkit['double']
            )
        else:
            new_editable_mol.AddBond(
                bond_atom_1,
                bond_atom_2,
                order=bond.GetBondType()
            )
    new_mol = new_editable_mol.GetMol()
    # print(Chem.MolToMolBlock(new_mol))

    new_mol.UpdatePropertyCache()
    # Chem.SanitizeMol(new_mol)
    return new_mol


def get_structures_from_mol(mol: Chem.Mol, dataset_cif_path, max_conformers):
    # Open the cif document with gemmi
    cif = gemmi.cif.read(str(dataset_cif_path))

    # Find the relevant atoms loop
    atom_id_loop = list(cif['comp_LIG'].find_loop('_chem_comp_atom.atom_id'))
    # print(f"Atom ID loop: {atom_id_loop}")


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
            # print(f"{j} : {atom_symbol}")

            # if atom_symbol == "H":
            #     continue
            gemmi_element: gemmi.Element = gemmi.Element(atom_symbol)

            # Get the position as a gemmi type
            pos: np.ndarray = positions[j, :]
            gemmi_pos: gemmi.Position = gemmi.Position(pos[0], pos[1], pos[2])

            # Get the
            gemmi_atom: gemmi.Atom = gemmi.Atom()
            # gemmi_atom.name = atom_symbol
            gemmi_atom.name = atom_id_loop[j]
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

    if ligand_files.ligand_cif is not None:
        mol = get_fragment_mol_from_dataset_cif_path(ligand_files.ligand_cif)

        # Generate conformers
        # mol.CalcImplicitValence()
        # mol: Chem.Mol = Chem.AddHs(mol)

        # Generate conformers
        cids = AllChem.EmbedMultipleConfs(
            mol,
            numConfs=num_pose_samples,
            pruneRmsThresh=pruning_threshold)

        # Translate to structures
        fragment_structures = get_structures_from_mol(
            mol,
            ligand_files.ligand_cif,
            max_conformers,
        )

        return fragment_structures


    # if ligand_files.ligand_smiles is not None:
    #     # print(f"\t\t\tGetting conformers from: {ligand_files.ligand_smiles}")
    #     mol = get_fragment_mol_from_dataset_smiles_path(ligand_files.ligand_smiles)
    #
    #     # Generate conformers
    #     mol: Chem.Mol = Chem.AddHs(mol)
    #
    #     # Generate conformers
    #     cids = AllChem.EmbedMultipleConfs(
    #         mol,
    #         numConfs=num_pose_samples,
    #         pruneRmsThresh=pruning_threshold)
    #
    #     # Translate to structures
    #     fragment_structures = get_structures_from_mol(
    #         mol,
    #         max_conformers,
    #     )
    #
    #     return fragment_structures

    if ligand_files.ligand_pdb is not None:
        # print(f"\t\t\tGetting conformers from: {ligand_files.ligand_pdb}")

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


def get_probe_structure(
        structure,
        bond_probes=False,
        neighbourhood_probes=False
):
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
    if bond_probes:
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

    if neighbourhood_probes:
        neighbourhood_probe_atoms = []
        for atom_index in verticies:

            atom = verticies[atom_index]
            pos = atom.pos
            for dx, dy, dz in itertools.product(
                [-0.4, 0.4],
                [-0.4, 0.4],
                [-0.4, 0.4],
            ):
                virtual_atom = gemmi.Atom()
                new_pos = gemmi.Position(
                    pos.x + dx,
                    pos.y + dy,
                    pos.z + dz,

                )
                atom_symbol: str = "C"
                virtual_atom.name = atom_symbol
                gemmi_element: gemmi.Element = gemmi.Element(atom_symbol)
                virtual_atom.element = gemmi_element
                virtual_atom.pos = new_pos
                neighbourhood_probe_atoms.append(virtual_atom)

        for model in structure_clone:
            for chain in model:
                for residue in chain:
                    for virtual_atom in neighbourhood_probe_atoms:
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
        degrees=True,
    )
    rotation_matrix: np.ndarray = rotation.as_matrix()

    transformed_structure_array = transform_structure_array(
        structure_array,
        np.array([x_2, y_2, z_2]),
        rotation_matrix
    )

    n = transformed_structure_array.shape[0]

    vals = get_interpolated_values_c(grid, transformed_structure_array, n)

    vals[vals > 3.0] = 3.0
    # vals[vals < 0.0] = 0.0

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
    probe_structure = get_probe_structure(
        centered_structure,
        neighbourhood_probes=True
    )

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

    total_evolve_time = 0.0
    time_begin_score = time.time()
    for j in range(event_fit_num_trys):
        # print(f"\t\t\t\tOptimizing round {j}")
        time_begin_evolve = time.time()
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
        # res = optimize.shgo(
        #     lambda params: score_fit_nonquant_array(
        #         structure_array,
        #         zmap_grid,
        #         1.0,
        #         params
        #     ),
        #     [
        #         (-6.0, 6.0), (-6, 6.0), (-6.0, 6.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)
        #     ],
        #     # sampling_method='sobol',
        #     # n=10000,
        #     iters=3
        #     # popsize=30,
        # )
        # res = optimize.basinhopping(
        #     lambda params: score_fit_nonquant_array(
        #         structure_array,
        #         zmap_grid,
        #         1.0,
        #         params
        #     ),
        #     [0.0,0.0,0.0,0.5,0.5,0.5],
        #     stepsize=0.05,
        #     niter=200
        #     # [
        #     #     (-6.0, 6.0), (-6, 6.0), (-6.0, 6.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)
        #     # ],
        #     # sampling_method='sobol',
        #     # n=10000,
        #     # iters=3
        #     # popsize=30,
        # )
        # print(res)
        time_finish_evolve = time.time()
        total_evolve_time += (time_finish_evolve-time_begin_evolve)
        # print(f"\t\t\t\t\tFinished Optimizing round {j}")

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
    time_finish_score = time.time()
    # print(f"\t\t\tScored conformer in {time_finish_score-time_begin_score} seconds, of which {total_evolve_time} evolving!")

    # print(f"Scores: {scores}")
    best_score_index = np.argmin(scores)

    best_score_fit_score = scores[best_score_index]
    best_optimised_structure = optimised_structures[best_score_index]

    return best_optimised_structure, float(best_score_fit_score), get_structure_mean(best_optimised_structure)


def get_score_grid(dmap, st, event: EventInterface):
    # Get a mask of the protein
    inner_mask_grid = gemmi.Int8Grid(dmap.nu, dmap.nv, dmap.nw)
    inner_mask_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    inner_mask_grid.set_unit_cell(dmap.unit_cell)

    ns = gemmi.NeighborSearch(st.structure[0], st.structure.cell, 12).populate(include_h=False)

    centroid = np.mean(event.pos_array, axis=0)

    centoid_pos = gemmi.Position(*centroid)
    marks = ns.find_atoms(centoid_pos, '\0', radius=11)

    for mark in marks:
        cra = mark.to_cra(st.structure[0])
        residue = cra.residue
        if residue.name in constants.RESIDUE_NAMES:
            # mark_pos = mark.pos
            # pos = gemmi.Position(mark_pos.x, mark_pos.y, mark_pos.z)
            pos = gemmi.Position(mark.x, mark.y, mark.z)
            inner_mask_grid.set_points_around(
                pos,
                radius=1.5,
                value=1,
            )
    # #
    # for model in st.structure:
    #     for chain in model:
    #         for residue in chain:
    #             if residue.name in constants.RESIDUE_NAMES:
    #                 for atom in residue:
    #                     pos = atom.pos
    #                     inner_mask_grid.set_points_around(pos,
    #                                                       radius=1.5,
    #                                                       value=1,
    #                                                       )

    inner_mask_grid_array = np.array(inner_mask_grid, copy=False)
    # print(inner_mask_grid_array.size)

    # Zero out density overlapping the protein
    dmap_array = np.array(dmap, copy=False)
    # non_zero_dmap_array = d
    # print(f"")
    structure_mask_indicies = np.nonzero(inner_mask_grid_array)
    # print(f"Mask indicies size: {inner_mask_grid_array[0].size}")
    dmap_array[structure_mask_indicies] = 0.0

    return dmap

def mask_dmap(dmap_array, st, reference_frame):
    dmap = reference_frame.unmask(SparseDMap(dmap_array))
    # Get a mask of the protein
    inner_mask_grid = gemmi.Int8Grid(dmap.nu, dmap.nv, dmap.nw)
    inner_mask_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    inner_mask_grid.set_unit_cell(dmap.unit_cell)

    for model in st:
        for chain in model:
            for residue in chain:
                if residue.name in constants.RESIDUE_NAMES:
                    for atom in residue:
                        pos = atom.pos
                        inner_mask_grid.set_points_around(pos,
                                                          radius=1.5,
                                                          value=1,
                                                          )

    inner_mask_grid_array = np.array(inner_mask_grid, copy=False)
    # print(inner_mask_grid_array.size)

    # Zero out density overlapping the protein
    dmap_array = np.array(dmap, copy=False)
    # non_zero_dmap_array = d
    # print(f"")
    structure_mask_indicies = np.nonzero(inner_mask_grid_array)
    # print(f"Mask indicies size: {inner_mask_grid_array[0].size}")
    dmap_array[structure_mask_indicies] = 0.0



    return SparseDMap.from_xmap(dmap, reference_frame).data


def get_event_grid(dmap, st, ):
    # Get a mask of the protein
    inner_mask_grid = gemmi.Int8Grid(dmap.nu, dmap.nv, dmap.nw)
    inner_mask_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    inner_mask_grid.set_unit_cell(dmap.unit_cell)

    # ns = gemmi.NeighborSearch(st.structure[0], st.structure.cell, 12).populate(include_h=False)

    # centroid = np.mean(event.pos_array, axis=0)
    #
    # centoid_pos = gemmi.Position(*centroid)
    # marks = ns.find_atoms(centoid_pos, '\0', radius=11)

    # for mark in marks:
    #     cra = mark.to_cra(st.structure[0])
    #     residue = cra.residue
    #     if residue.name in constants.RESIDUE_NAMES:
    #         # mark_pos = mark.pos
    #         # pos = gemmi.Position(mark_pos.x, mark_pos.y, mark_pos.z)
    #         pos = gemmi.Position(mark.x, mark.y, mark.z)
    #         inner_mask_grid.set_points_around(
    #             pos,
    #             radius=1.5,
    #             value=1,
    #         )
    #
    for model in st.structure:
        for chain in model:
            for residue in chain:
                if residue.name in constants.RESIDUE_NAMES:
                    for atom in residue:
                        pos = atom.pos
                        inner_mask_grid.set_points_around(pos,
                                                          radius=1.5,
                                                          value=1,
                                                          )

    inner_mask_grid_array = np.array(inner_mask_grid, copy=False)
    # print(inner_mask_grid_array.size)

    # Zero out density overlapping the protein
    dmap_array = np.array(dmap, copy=False)
    # non_zero_dmap_array = d
    # print(f"")
    structure_mask_indicies = np.nonzero(inner_mask_grid_array)
    # print(f"Mask indicies size: {inner_mask_grid_array[0].size}")
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
        # st = Structure.from_path(model_path)
        # st = dataset.structure
        st = Structure.from_path(str(dataset.structure.path))

        # Get the scoring grid
        dmap = load_dmap(dmap_path)
        score_grid = get_score_grid(dmap, st, event)

        # save_dmap(score_grid, out_dir / "score_grid.ccp4")

        # ligand_scoring_results = {}
        # for ligand_key, ligand_files in dataset.ligand_files.items():

        # Generate conformers to score
        # print(f"\tGetting conformermers!")
        conformers = get_conformers(ligand_files)
        # print(f"\t\tGot {len(conformers)} conformers!")

        if len(conformers) == 0:
            return

        # Score conformers against the grid
        conformer_scores = {}
        for conformer_id, conformer in conformers.items():
            optimized_structure, score, centroid = score_conformer(
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

def get_local_signal_dencalc(optimized_structure, event_map_grid, res, ):
    # Get the electron density of the optimized structure
    optimized_structure.cell = event_map_grid.unit_cell
    optimized_structure.spacegroup_hm = gemmi.find_spacegroup_by_name("P 1").hm
    dencalc = gemmi.DensityCalculatorE()
    dencalc.d_min = res#*2
    dencalc.rate = 2.0
    # initial_dencalc_grid = gemmi.FloatGrid(event_map_grid.nu, event_map_grid.nv, event_map_grid.nw)
    # initial_dencalc_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    # initial_dencalc_grid.set_unit_cell(event_map_grid.unit_cell)
    # dencalc.grid = initial_dencalc_grid
    dencalc.set_grid_cell_and_spacegroup(optimized_structure)
    dencalc.put_model_density_on_grid(optimized_structure[0])
    # dencalc.add_model_density_to_grid(optimized_structure[0])
    calc_grid = dencalc.grid
    calc_grid_array = np.array(calc_grid, copy=False)
    # print([event_map_grid.nu, event_map_grid.nv, event_map_grid.nw, calc_grid.nu, calc_grid.nv, calc_grid.nw])
    # print([calc_grid.nu, event_map_grid.nu])

    # Get the mask around the structure
    inner_mask_grid = gemmi.Int8Grid(event_map_grid.nu, event_map_grid.nv, event_map_grid.nw)
    inner_mask_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    inner_mask_grid.set_unit_cell(event_map_grid.unit_cell)

    num_atoms = 0
    for model in optimized_structure:
        for chain in model:
            for residue in chain:
                # if residue.name in constants.RESIDUE_NAMES:
                for atom in residue:
                    if atom.element.name=="H":
                        continue
                    pos = atom.pos
                    inner_mask_grid.set_points_around(pos,
                                                      radius=1.5,
                                                      value=1,
                                                      )
                    inner_mask_grid.set_points_around(pos,
                                                      radius=1.0,
                                                      value=2,
                                                      )
                    # inner_mask_grid.set_points_around(pos,
                    #                                   radius=1.0,
                    #                                   value=2,
                    #                                   )
                    inner_mask_grid.set_points_around(pos,
                                                      radius=0.75,
                                                      value=3,
                                                      )
                    num_atoms += 1

    inner_mask_grid_array = np.array(inner_mask_grid, copy=False)

    # Get the correlation with the event
    event_map_grid_array = np.array(event_map_grid, copy=False)
    masked_event_map_vals = event_map_grid_array[inner_mask_grid_array >= 2]
    masked_calc_vals = calc_grid_array[inner_mask_grid_array >= 2]
    corr = np.corrcoef(
        np.concatenate(
            (
                masked_event_map_vals.reshape(-1,1),
                masked_calc_vals.reshape(-1, 1)
            ),
            axis=1,
        )
    )[0,1]

    num_atoms = np.log(num_atoms)

    return corr #* num_atoms

def get_correlation(_bdc, masked_xmap_vals, masked_mean_map_vals, masked_calc_vals):

    event_map_vals = (masked_xmap_vals - (_bdc*masked_mean_map_vals)) / (1-_bdc)
    masked_event_map_vals = event_map_vals[event_map_vals != 0]
    double_masked_calc_vals = masked_calc_vals[event_map_vals != 0]

    if masked_event_map_vals.size == 0:
        return 1.0

    # corr = np.corrcoef(
    #     np.concatenate(
    #         (
    #             masked_event_map_vals.reshape(-1, 1),
    #             masked_calc_vals.reshape(-1, 1)
    #         ),
    #         axis=1,
    #     )
    # )[0, 1]
    # return 1-corr

    event_map_mean = np.mean(masked_event_map_vals)
    calc_map_mean = np.mean(double_masked_calc_vals)
    delta_event_map = masked_event_map_vals - event_map_mean
    delta_calc_map = double_masked_calc_vals-calc_map_mean
    nominator = np.sum(delta_event_map*delta_calc_map)
    denominator = np.sqrt(
        np.sum(np.square(delta_event_map))*np.sum(np.square(delta_calc_map))
    )

    corr = nominator / denominator

    return 1-corr

def get_predicted_mask(
        optimized_structure,
        xmap,
):
    # Get the mask around the structure
    inner_mask_grid = gemmi.Int8Grid(xmap.nu, xmap.nv, xmap.nw)
    inner_mask_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    inner_mask_grid.set_unit_cell(xmap.unit_cell)

    num_atoms = 0
    for model in optimized_structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element.name == "H":
                        continue
                    pos = atom.pos
                    inner_mask_grid.set_points_around(pos,
                                                      radius=1.5,
                                                      value=1,
                                                      )
                    inner_mask_grid.set_points_around(pos,
                                                      radius=1.0,
                                                      value=2,
                                                      )
                    # inner_mask_grid.set_points_around(pos,
                    #                                   radius=1.0,
                    #                                   value=2,
                    #                                   )
                    inner_mask_grid.set_points_around(pos,
                                                      radius=0.75,
                                                      value=3,
                                                      )
                    inner_mask_grid.set_points_around(pos,
                                                      radius=0.5,
                                                      value=4,
                                                      )
                    num_atoms += 1

    return inner_mask_grid

def get_predicted_density(
    optimized_structure,
    xmap
):
    # Get the electron density of the optimized structure
    optimized_structure.cell = xmap.unit_cell
    optimized_structure.spacegroup_hm = gemmi.find_spacegroup_by_name("P 1").hm
    dencalc = gemmi.DensityCalculatorE()
    # dencalc.d_min = res#*2
    # dencalc.rate = 2.0
    dencalc.set_grid_cell_and_spacegroup(optimized_structure)
    dencalc.initialize_grid_to_size(xmap.nu, xmap.nv, xmap.nw)
    dencalc.add_model_density_to_grid(optimized_structure[0])
    # dencalc.put_model_density_on_grid(optimized_structure[0])
    calc_grid = dencalc.grid
    # calc_grid_array = np.array(calc_grid, copy=False)

    return calc_grid

def get_predicted_density_high_contour(
        predicted_density,
        predicted_mask
    ):
    predicted_density_array = np.array(predicted_density, copy=False)
    predicted_mask_array = np.array(predicted_mask, copy=False)

    predicted_high_vals = predicted_density_array[predicted_mask_array >3]

    contour = np.quantile(
        predicted_high_vals,
        0.0
    )

    return contour

def get_local_signal_dencalc_optimize_bdc(
        optimized_structure,
        calc_grid,
        predicted_mask,
        reference_frame,
        dtag_vals,
        mean_vals,
        res,
        event_bdc,
):
    # Get the unmasked xmap and mean map
    xmap = reference_frame.unmask(dtag_vals)
    mean_map = reference_frame.unmask(mean_vals)
    xmap_array = np.array(xmap, copy=False)
    mean_map_array = np.array(mean_map, copy=False)

    calc_grid_array = np.array(calc_grid, copy=False)

    inner_mask_grid_array = np.array(predicted_mask, copy=False)

    # Pull out the ligand masked xmap and mean map vals
    masked_xmap_vals = xmap_array[inner_mask_grid_array >= 2]
    masked_mean_map_vals = mean_map_array[inner_mask_grid_array >= 2]
    masked_calc_vals = calc_grid_array[inner_mask_grid_array >= 2]

    # res = optimize.minimize(
    #     lambda _bdc: get_correlation(_bdc, masked_xmap_vals, masked_mean_map_vals, masked_calc_vals),
    #     event_bdc,
    #     bounds=((0.0, 0.95),),
    #     # tol=0.1
    # )
    res = optimize.differential_evolution(
        lambda _bdc: get_correlation(
            _bdc,
            masked_xmap_vals,
            masked_mean_map_vals,
            masked_calc_vals,
        ),
        [(0.0, 0.95), ],
        # popsize=30,
    )

    # # Get the correlation with the event
    # event_map_grid_array = np.array(event_map_grid, copy=False)
    # masked_event_map_vals = event_map_grid_array[inner_mask_grid_array >= 2]
    # masked_calc_vals = calc_grid_array[inner_mask_grid_array >= 2]
    # corr = np.corrcoef(
    #     np.concatenate(
    #         (
    #             masked_event_map_vals.reshape(-1,1),
    #             masked_calc_vals.reshape(-1, 1)
    #         ),
    #         axis=1,
    #     )
    # )[0,1]

    # num_atoms = np.log(num_atoms)
    bdc = res.x
    corr = 1-res.fun

    # masked_event_map_vals = (masked_xmap_vals - (bdc * masked_mean_map_vals)) / (1 - bdc)

    # print(f"Refined to bdc: {bdc} and correlation: {corr} on set of size: {masked_xmap_vals.size}")

    return corr, bdc #* num_atoms

def get_local_signal(optimized_structure, event_map_grid):
    event_map_grid_array = np.array(event_map_grid, copy=False)

    event_map_grid_array[event_map_grid_array < 0.0] = 0.0
    event_map_grid_array[event_map_grid_array > 2.0] = 2.0
    inner_mask_grid = gemmi.Int8Grid(event_map_grid.nu, event_map_grid.nv, event_map_grid.nw)
    inner_mask_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    inner_mask_grid.set_unit_cell(event_map_grid.unit_cell)

    for model in optimized_structure:
        for chain in model:
            for residue in chain:
                # if residue.name in constants.RESIDUE_NAMES:
                for atom in residue:
                    if atom.element.name == "H":
                        continue
                    pos = atom.pos
                    inner_mask_grid.set_points_around(pos,
                                                      radius=1.5,
                                                      value=1,
                                                      )
                    inner_mask_grid.set_points_around(pos,
                                                      radius=1.0,
                                                      value=2,
                                                      )
                    # inner_mask_grid.set_points_around(pos,
                    #                                   radius=1.0,
                    #                                   value=2,
                    #                                   )
                    inner_mask_grid.set_points_around(pos,
                                                      radius=0.75,
                                                      value=3,
                                                      )

    inner_mask_grid_array = np.array(inner_mask_grid, copy=False)

    # vals_pos = event_map_grid_array[np.nonzero(inner_mask_grid_array == 2)]
    # vals_neg = event_map_grid_array[np.nonzero(inner_mask_grid_array == 1)]
    full_mask = event_map_grid_array[np.nonzero(inner_mask_grid_array >= 1)]
    background = np.mean(full_mask)

    outer_mask = event_map_grid_array[np.nonzero(inner_mask_grid_array == 1)]
    outer_mean = np.mean(outer_mask)
    outer_std = np.std(outer_mean)
    # background = np.mean(outer_mask)
    background = outer_mean + (2*outer_std)

    # high_non_core = np.sum(outer_mask > background)
    # low_non_core = np.sum(outer_mask <= background)
    # non_core_score = low_non_core-high_non_core
    # non_core_score = low_non_core * (low_non_core / outer_mask.size)

    core_points = event_map_grid_array[np.nonzero(inner_mask_grid_array == 3)]
    high_core = np.sum(core_points > background)
    # low_core = np.sum(core_points <= background)
    # core_score = high_core-low_core
    # core_score = high_core * (high_core/core_points.size)

    # score = core_score+(non_core_score*(core_points.size / outer_mask.size))
    score = high_core

    # return np.sum(vals_pos-np.mean(vals_neg)) #- np.sum(vals_neg)
    return score

def get_signal(
        xmap_array,
        xmap_mask
):
    return xmap_array[xmap_mask]

def get_optimal_signal_contour(
        signal_vals,
        noise_signal_vals
    ):
    diffs = {}
    for val in np.linspace(
        np.min(signal_vals),
        np.max(signal_vals),
        num=100
    ):
        difference = int(np.sum(signal_vals > val) - np.sum(noise_signal_vals > val))
        diffs[float(val)] = difference

    return max(diffs, key=lambda _key: diffs[_key])

def get_contacts(
        optimized_structure,
        st
    ):
    ns = gemmi.NeighborSearch(st[0], st.cell, 5).populate(include_h=False)

    contacts = []
    for model in optimized_structure:
        for chain in model:
            for res in chain:
                for atom in res:
                    # contacts = []
                    marks = ns.find_atoms(atom.pos, '\0', radius=3.2)
                    for mark in marks:
                        cra = mark.to_cra(st[0])
                        res = cra.residue
                        if res.name in constants.RESIDUE_NAMES:
                            contacts.append(res.seqid.num)
    num_contacts = len(set(contacts))
    return num_contacts
    ...


def autobuild_conformer(
        centroid,
        event_bdc,
        conformer,
        masked_dtag_array,
        masked_mean_array,
        reference_frame,
        out_dir,
        conformer_id,
        res,
        structure,
        unmasked_dtag_array,
        unmasked_mean_array,
        z_array,
    raw_xmap_sparse,
    score_build
                ):

    time_begin_autobuild = time.time()

    # event_map_grid = reference_frame.unmask(SparseDMap((masked_dtag_array - (event_bdc*masked_mean_array)) / (1-event_bdc)))
    event_map_array = (masked_dtag_array - (event_bdc*masked_mean_array)) / (1-event_bdc)
    normalize_event_map_array = (event_map_array-np.mean(event_map_array) / np.std(event_map_array))
    normalize_z_array = (z_array - np.mean(z_array)) / np.std(z_array)
    score_grid = normalize_z_array + normalize_event_map_array
    event_map_grid = reference_frame.unmask(SparseDMap(score_grid))
    z_grid = reference_frame.unmask(SparseDMap(z_array))
    raw_xmap_grid = reference_frame.unmask(SparseDMap(raw_xmap_sparse))

    optimized_structure, score, centroid = score_conformer(
        centroid,
        conformer.structure,
        event_map_grid,
    )

    save_structure(
        Structure(None, optimized_structure),
        out_dir / f"{conformer_id}.pdb",
    )

    xmap = reference_frame.unmask(masked_dtag_array)

    predicted_mask = get_predicted_mask(
        optimized_structure,
        xmap
    )
    predicted_mask_array = np.array(predicted_mask, copy=False)

    predicted_density = get_predicted_density(
        optimized_structure,
        xmap
    )
    predicted_density_array = np.array(predicted_density, copy=False)

    try:
        predicted_density_high_contour = get_predicted_density_high_contour(
            predicted_density,
            predicted_mask
        )
    except:
        predicted_density_high_contour = 1.0

    corr, bdc = get_local_signal_dencalc_optimize_bdc(
        optimized_structure,
        predicted_density,
        predicted_mask,
        reference_frame,
        unmasked_dtag_array,
        unmasked_mean_array,
        res, event_bdc
    )

    corrected_event_map_grid = reference_frame.unmask(SparseDMap((unmasked_dtag_array - (bdc*unmasked_mean_array)) / (1-bdc)))
    corrected_event_map_array = np.array(corrected_event_map_grid, copy=False)

    signal_vals = get_signal(
        corrected_event_map_array,
        predicted_density_array > predicted_density_high_contour
    )
    # signal_z = ...

    noise_signal_vals = get_signal(
        corrected_event_map_array,
        predicted_mask_array == 1
    )
    # noise_singal_z = ...

    try:
        optimal_signal_contour = get_optimal_signal_contour(
            signal_vals,
            noise_signal_vals
        )
    except:
        optimal_signal_contour = 1.0

    optimal_noise = np.sum(noise_signal_vals > optimal_signal_contour)
    optimal_signal = np.sum(signal_vals > optimal_signal_contour)

    num_contacts = get_contacts(
        optimized_structure,
        structure.structure
    )

    noise_signal_vals[noise_signal_vals <0] = 0.0
    noise_signal_vals[noise_signal_vals > 3] = 3.0
    signal_vals[signal_vals < 0] = 0.0
    signal_vals[signal_vals > 3] = 3.0

    time_begin_scoring = time.time()
    score = score_build(
        optimized_structure,
        # corrected_event_map_grid,
        reference_frame.unmask(SparseDMap(unmasked_dtag_array)),
        reference_frame.unmask(SparseDMap(unmasked_mean_array)),
        bdc,
        z_grid,
        raw_xmap_grid,
    )
    time_finish_scoring = time.time()

    log_result_dict = {
        str(out_dir / f"{conformer_id}.pdb"): {
        'score': score,
        'centroid': centroid,
        # 'local_signal': get_local_signal(optimized_structure, event_map_grid)
        # 'local_signal': get_local_signal_dencalc(
        #     optimized_structure,
        #     event_map_grid,
        #     res,
        # )
        'local_signal': float(corr),
        'new_bdc': float(bdc),
        # 'noise': float(noise_signal),
        # 'signal': float(signal),
        # 'noise': float(np.abs(np.median(noise_signal_vals))),
        # 'signal': float(np.abs(np.median(signal_vals))),
            'noise': float(np.abs(np.sum(noise_signal_vals))),
            'signal': float(np.abs(np.sum(signal_vals))),
        'num_points': int(np.sum(predicted_density_array > predicted_density_high_contour)),
            'optimal_contour': float(optimal_signal_contour),
        'num_contacts': int(num_contacts),
            # 'total_noise':
        }
    }
    time_finish_autobuild = time.time()
    print(f"Scored in: {time_finish_scoring-time_begin_scoring}")

    # Return results
    return log_result_dict


class AutobuildModelEventInbuilt:

    def __init__(self, cut=2.0):
        self.cut = cut

    def __call__(
            self,
            event: EventInterface,
            dataset: DatasetInterface,
            dmap,
            mtz,
            model,
            ligand_files,
            out_dir,
    ):

        # Get the structure
        st = Structure.from_path(str(dataset.structure.path))

        # Get the scoring grid
        score_grid = get_score_grid(dmap, st, event)

        # Generate conformers to score
        conformers = get_conformers(ligand_files)
        if len(conformers) == 0:
            return AutobuildResult(
                {},
                None,
                None,
                None,
                ligand_files.ligand_cif,
                out_dir
            )

        # Score conformers against the grid
        conformer_scores = {}
        for conformer_id, conformer in conformers.items():
            optimized_structure, score, centroid = score_conformer(
                np.mean(event.pos_array, axis=0),
                conformer,
                score_grid,
            )
            conformer_scores[conformer_id] = [optimized_structure, score, centroid]

        if len(conformer_scores) == 0:
            return AutobuildResult(
                {},
                None,
                None,
                None,
                ligand_files.ligand_cif,
                out_dir
            )

        # Choose the best ligand
        for conformer_id, (optimized_structure, score, centroid) in conformer_scores.items():
            save_structure(
                Structure(None, optimized_structure),
                out_dir / f"{conformer_id}.pdb",
            )

        log_result_dict = {
            str(out_dir / f"{conformer_id}.pdb"): {'score': score,
            'centroid': centroid,}
            for conformer_id, (optimized_structure, score, centroid)
            in conformer_scores.items()
        }

        # Return results
        return AutobuildResult(
            log_result_dict,
            None,
            None,
            None,
            ligand_files.ligand_cif,
            out_dir
        )
