# 3rd party
import numpy as np
import gemmi
from rdkit import Chem
from rdkit.Chem import AllChem
import joblib
import scipy
from scipy import spatial as spsp, optimize
from pathlib import Path
import time

#
from pandda_gemmi.analyse_interface import *
from pandda_gemmi.dataset import Dataset
# from pandda_gemmi.fs import PanDDAFSModel, ProcessedDataset
from pandda_gemmi.event import Cluster
# from pandda_gemmi.autobuild import score_structure_signal_to_noise_density, EXPERIMENTAL_score_structure_signal_to_noise_density
from pandda_gemmi.scoring import EXPERIMENTAL_score_structure_signal_to_noise_density
from pandda_gemmi.python_types import *


class ConformerFittingResult(ConformerFittingResultInterface):
    def __init__(self,
                 score: Optional[float],
                 optimised_fit: Optional[Any],
                 score_log: Optional[Dict]
                 ):
        self.score: Optional[float] = score
        self.optimised_fit: Optional[Any] = optimised_fit
        self.score_log = score_log

    def log(self) -> Dict:
        return {
            "Score": str(self.score),
            "Score Log": self.score_log
        }

    def __setstate__(self, state):
        self.score = state[0]

        if state[1] is not None:
            self.optimised_fit = state[1].to_gemmi()
            self.optimised_fit.setup_entities()
        else:
            self.optimised_fit = state[1]

        self.score_log = state[2]

    def __getstate__(self):
        if self.optimised_fit is not None:
            return (self.score, StructurePython.from_gemmi(self.optimised_fit), self.score_log)
        else:
            return (self.score, self.optimised_fit, self.score_log)


def get_structures_from_mol(mol: Chem.Mol, max_conformers) -> MutableMapping[int, gemmi.Structure]:
    fragment_structures: MutableMapping[int, gemmi.Structure] = {}
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


def get_fragment_mol_from_dataset_smiles_path(dataset_smiles_path: Path):
    smiles_path = dataset_smiles_path

    # Get smiels string
    with open(str(smiles_path), "r") as f:
        smiles_string: str = str(f.read())

    # Load the mol
    m: Chem.Mol = Chem.MolFromSmiles(smiles_string)

    return m


def structure_from_small_structure(small_structure):
    structure: gemmi.Structure = gemmi.Structure()
    model: gemmi.Model = gemmi.Model(f"0")
    chain: gemmi.Chain = gemmi.Chain(f"0")
    residue: gemmi.Residue = gemmi.Residue()
    residue.name = "LIG"
    residue.seqid = gemmi.SeqId(1, ' ')

    # Loop over atoms, adding them to a gemmi residue
    for j, site in enumerate(small_structure.sites):
        # Get the atomic symbol
        # atom_symbol: str = site.element
        gemmi_element: gemmi.Element = site.element

        # Get the position as a gemmi type
        # pos: np.ndarray = positions[j, :]
        gemmi_pos: gemmi.Position = site.orth(small_structure.cell)

        # Get the
        gemmi_atom: gemmi.Atom = gemmi.Atom()
        gemmi_atom.name = site.label
        gemmi_atom.pos = gemmi_pos
        gemmi_atom.element = gemmi_element

        # Add atom to residue
        residue.add_atom(gemmi_atom)

    chain.add_residue(residue)
    model.add_chain(chain)
    structure.add_model(model)

    return structure


def structures_from_cif(source_ligand_cif, debug: Debug = Debug.DEFAULT):
    # doc = gemmi.cif.read_file(str(source_ligand_cif))
    # block = doc[-1]
    # cc = gemmi.make_chemcomp_from_block(block)
    # cc.remove_hydrogens()
    # small_structure = gemmi.read_small_structure(str(source_ligand_cif))
    cif_doc = gemmi.cif.read(str(source_ligand_cif))
    # for block in cif_doc:
    small_structure = gemmi.make_small_structure_from_block(cif_doc[-1])
    if debug >= Debug.PRINT_NUMERICS:
        print(f"\t\t\tsmall_structure: {small_structure}")
        print(f"\t\t\tSmall structure sites: {small_structure.sites}")

    if len(small_structure.sites) == 0:
        return {}

    structure = structure_from_small_structure(small_structure)

    # for atom in cc.atoms:
    #     G.add_node(atom.id, Z=atom.el.atomic_number)
    # for bond in cc.rt.bonds:
    #     G.add_edge(bond.id1.atom, bond.id2.atom)
    return {0: structure}


class Conformers(ConformersInterface):

    def __init__(self,
                 conformers: Dict[int, Any],
                 method: str,
                 path: Optional[Path],
                 ):
        self.conformers: Dict[int, Any] = conformers
        self.method: str = method
        self.path: Optional[Path] = path

    def log(self) -> Dict:
        return {
            "Num conformers": str(len(self.conformers)),
            "Ligand generation method": self.method,
            "Ligand source path": str(self.path)
        }

    def __getstate__(self):
        return (
            {key: StructurePython.from_gemmi(value) for key, value in self.conformers.items()},
            self.method,
            self.path
        )

    def __setstate__(self, state):
        self.conformers = {key: value.to_gemmi() for key, value in state[0].items()}
        self.method = state[1]
        self.path = state[2]


def get_conformers(
        fragment_dataset,
        pruning_threshold=2.0,
        num_pose_samples=1000,
        max_conformers=10,
        debug: Debug = Debug.DEFAULT,
) -> ConformersInterface:
    # Decide how to load
    # fragment_structures = {}
    if fragment_dataset.source_ligand_smiles:

        if debug >= Debug.PRINT_NUMERICS:
            print(f'\t\tGetting mol from ligand smiles')
        mol = get_fragment_mol_from_dataset_smiles_path(fragment_dataset.source_ligand_smiles)

        # Generate conformers
        mol: Chem.Mol = Chem.AddHs(mol)

        # Generate conformers
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=num_pose_samples, pruneRmsThresh=pruning_threshold)

        # Translate to structures
        fragment_structures: MutableMapping[int, gemmi.Structure] = get_structures_from_mol(mol, max_conformers)
        if len(fragment_structures) > 0:
            return Conformers(
                fragment_structures,
                "smiles",
                fragment_dataset.source_ligand_smiles,
            )

    if fragment_dataset.source_ligand_cif:
        if debug >= Debug.PRINT_NUMERICS:
            print(f'\t\tGetting mol from cif')
        fragment_structures = structures_from_cif(fragment_dataset.source_ligand_cif, debug)
        if len(fragment_structures) > 0:
            return Conformers(
                fragment_structures,
                "cif",
                fragment_dataset.source_ligand_cif,
            )

    if fragment_dataset.source_ligand_pdb:
        if debug >= Debug.PRINT_NUMERICS:
            print(f'\t\tGetting mol from ligand pdb')
        fragment_structures = {0: gemmi.read_structure(str(fragment_dataset.source_ligand_pdb))}
        if len(fragment_structures) > 0:
            return Conformers(
                fragment_structures,
                "pdb",
                fragment_dataset.source_ligand_pdb,
            )

    if fragment_dataset.source_ligand_smiles or fragment_dataset.source_ligand_cif or \
            fragment_dataset.source_ligand_pdb:
        if debug >= Debug.PRINT_NUMERICS:
            print(fragment_structures)

    return Conformers(
        {},
        "None",
        None,
    )


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


def score_fit(structure, grid, distance, params):
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

    structure_copy = transform_structure(
        structure,
        [x_2, y_2, z_2],
        rotation_matrix
    )

    vals = []
    n = 0
    for model in structure_copy:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element.name != "H":
                        vals.append(
                            grid.interpolate_value(
                                atom.pos
                            )
                        )
                        n = n + 1

    positive_score = sum([1 if val > 0.5 else 0 for val in vals])
    penalty = sum([-1 if val < -0.0 else 0 for val in vals])
    score = (positive_score + penalty) / n

    # return 1 - (sum([1 if val > 2.0 else 0 for val in vals ]) / n)
    return 1 - score


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


def get_interpolated_values(grid, transformed_structure_array):
    vals = []
    for row in transformed_structure_array:
        pos = gemmi.Position(row[0], row[1], row[2])
        vals.append(
            grid.interpolate_value(
                pos
            )
        )

    return np.array(vals)


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


def score_fit_array(structure_array, grid, distance, params):
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
    # print(f"Interpolated vals: {vals}: {np.sum(vals)}")
    # print(f"Num points > 2.0: {np.sum(np.array(grid) == 1.0)}")
    # print(f"Transformed structure array: {transformed_structure_array}")

    # print(type(transformed_structure_array))
    # vals = grid.interpolate_values_from_pos_array(transformed_structure_array)

    positive_score = np.sum(vals > 0.5)
    penalty = -np.sum(vals < 0.0)
    score = (positive_score + penalty) / n

    return float(1 - score)


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
    # print(f"Interpolated vals: {vals}: {np.sum(vals)}")
    # print(f"Num points > 2.0: {np.sum(np.array(grid) == 1.0)}")
    # print(f"Transformed structure array: {transformed_structure_array}")

    # print(type(transformed_structure_array))
    # vals = grid.interpolate_values_from_pos_array(transformed_structure_array)

    # positive_score = np.sum(vals > 0.5)
    # penalty = -np.sum(vals < 0.0)
    # score = (positive_score + penalty) / n
    vals[vals > 3.0] = 3.0

    score = np.sum(vals)

    return float(-score)


def DEP_score_fit(structure, grid, distance, params):
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
    structure_copy = transform_structure(
        structure,
        [x_2, y_2, z_2],
        rotation_matrix
    )

    vals = []
    n = 0
    for model in structure_copy:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element.name != "H":
                        vals.append(
                            grid.interpolate_value(
                                atom.pos
                            )
                        )
                        n = n + 1

    # positive_score = sum([1 if val > 0.5 else 0 for val in vals])
    # penalty = sum([-1 if val < -0.0 else 0 for val in vals])
    # score = (positive_score + penalty) / n

    score, log = EXPERIMENTAL_score_structure_signal_to_noise_density(
        structure_copy,
        grid,
    )

    # return 1 - (sum([1 if val > 2.0 else 0 for val in vals ]) / n)
    return score


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


def EXPERIMENTAL_score_structure_rscc(
        optimised_structure,
        zmap_grid,
        res,
        rate,
) -> Tuple[float, Dict]:
    # Get grid
    new_grid = gemmi.FloatGrid(zmap_grid.nu, zmap_grid.nv, zmap_grid.nw)
    new_grid.unit_cell = zmap_grid.unit_cell

    # Get density on grid
    optimised_structure.spacegroup_hm = gemmi.find_spacegroup_by_name("P 1").hm
    optimised_structure.cell = zmap_grid.unit_cell
    dencalc = gemmi.DensityCalculatorX()
    dencalc.d_min = 0.5
    dencalc.rate = 1
    dencalc.set_grid_cell_and_spacegroup(optimised_structure)
    dencalc.put_model_density_on_grid(optimised_structure[0])

    # Get the SFs at the right res
    sf = gemmi.transform_map_to_f_phi(dencalc.grid, half_l=True)
    data = sf.prepare_asu_data(dmin=res)

    # Get the grid oversampled to the right size
    approximate_structure_map = data.transform_f_phi_to_map(exact_size=[zmap_grid.nu, zmap_grid.nv, zmap_grid.nw])

    # Get mask of ligand
    inner_mask_grid = gemmi.Int8Grid(*[zmap_grid.nu, zmap_grid.nv, zmap_grid.nw])
    inner_mask_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    inner_mask_grid.set_unit_cell(zmap_grid.unit_cell)
    for struc in optimised_structure:
        for chain in struc:
            for res in chain:
                for atom in res:
                    pos = atom.pos
                    inner_mask_grid.set_points_around(pos,
                                                      radius=1.0,
                                                      value=1,
                                                      )

    outer_mask_grid = gemmi.Int8Grid(*[zmap_grid.nu, zmap_grid.nv, zmap_grid.nw])
    outer_mask_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    outer_mask_grid.set_unit_cell(zmap_grid.unit_cell)
    for struc in optimised_structure:
        for chain in struc:
            for res in chain:
                for atom in res:
                    pos = atom.pos
                    outer_mask_grid.set_points_around(pos,
                                                      radius=2.0,
                                                      value=1,
                                                      )
    for struc in optimised_structure:
        for chain in struc:
            for res in chain:
                for atom in res:
                    pos = atom.pos
                    outer_mask_grid.set_points_around(pos,
                                                      radius=1.0,
                                                      value=0,
                                                      )
    outer_mask_array = np.array(outer_mask_grid, copy=False, dtype=np.int8, )
    outer_mask_indexes = np.nonzero(outer_mask_array)

    # Scale ligand density to map
    inner_mask_int_array = np.array(
        inner_mask_grid,
        copy=False,
        dtype=np.int8,
    )
    event_map_array = np.array(zmap_grid, copy=False)
    approximate_structure_map_array = np.array(approximate_structure_map, copy=False)
    mask_indicies = np.nonzero(inner_mask_int_array)

    event_map_values = event_map_array[mask_indicies]
    approximate_structure_map_values = approximate_structure_map_array[mask_indicies]

    # scaled_event_map_values = ((approximate_structure_map_values - np.mean(approximate_structure_map_values))
    #                           * (np.std(event_map_values) / np.std(approximate_structure_map_values))) \
    #                           + (np.mean(event_map_values))
    #

    # Get correlation
    demeaned_event_map_values = event_map_values - np.mean(event_map_values)
    demeaned_approximate_structure_map_values = approximate_structure_map_values - np.mean(
        approximate_structure_map_values)
    corr = np.sum(demeaned_event_map_values * demeaned_approximate_structure_map_values
                  ) / (
                   np.sqrt(np.sum(np.square(demeaned_event_map_values)))
                   * np.sqrt(np.sum(np.square(demeaned_approximate_structure_map_values)))
           )

    scores = {}
    for cutoff in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.8, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]:
        noise_percent = np.sum(event_map_array[outer_mask_indexes] > cutoff) / np.sum(outer_mask_array)
        signal = np.sum(event_map_array[mask_indicies] > cutoff)
        score = signal - (np.sum(inner_mask_int_array) * noise_percent)
        scores[float(cutoff)] = int(score)

    scores_from_calc = {}
    for cutoff in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.8, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]:
        approximate_structure_map_high_indicies = approximate_structure_map_array > 1.5
        event_map_high_indicies = event_map_array[approximate_structure_map_high_indicies] > cutoff
        signal = np.sum(event_map_high_indicies)
        noise_percent = np.sum(event_map_array[outer_mask_indexes] > cutoff) / np.sum(outer_mask_array)
        score = signal - (np.sum(approximate_structure_map_array > 1.5) * noise_percent)
        scores_from_calc[float(cutoff)] = int(score)

    score = max(scores_from_calc.values())

    return score, {
        "Num masked indicies": len(mask_indicies[0]),
        "Mean approximate density": float(np.mean(approximate_structure_map_values)),
        "Mean event density": float(np.mean(event_map_values)),
        "grid": approximate_structure_map,
        "outer_sum": float(np.sum(event_map_array[outer_mask_indexes])),
        "inner_sum": float(np.sum(event_map_array[mask_indicies])),
        "outer_mean": float(np.mean(event_map_array[outer_mask_indexes])),
        "inner_mean": float(np.mean(event_map_array[mask_indicies])),
        "inner>1": int(np.sum(event_map_array[mask_indicies] > 1.0)),
        "outer>1": int(np.sum(event_map_array[outer_mask_indexes] > 1.0)),
        "inner>2": int(np.sum(event_map_array[mask_indicies] > 2.0)),
        "outer>2": int(np.sum(event_map_array[outer_mask_indexes] > 2.0)),
        "num_inner": int(np.sum(inner_mask_int_array)),
        "num_outer": int(np.sum(outer_mask_array)),
        "scores": scores,
        "scores_from_calc": scores_from_calc
    }


def score_conformer_nonquant_array(cluster: Cluster,
                                   conformer,
                                   zmap_grid,
                                   resolution,
                                   rate,
                                   debug: Debug = Debug.DEFAULT) -> ConformerFittingResultInterface:
    # Center the conformer at the cluster
    centroid_cart = cluster.centroid

    if debug >= Debug.PRINT_NUMERICS:
        print(f"\t\t\t\tCartesian centroid of event is: {centroid_cart}")

    centered_structure = center_structure(
        conformer,
        centroid_cart,
    )

    # Get the probe structure
    probe_structure = get_probe_structure(centered_structure)

    if debug >= Debug.PRINT_NUMERICS:
        print(f"\t\t\t\tprobe structure: {probe_structure}")

    # Optimise
    if debug >= Debug.PRINT_NUMERICS:
        print(f"\t\t\t\tOptimizing structure fit...")

    structure_positions = []

    for model in probe_structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element.name != "H":
                        pos = atom.pos
                        structure_positions.append([pos.x, pos.y, pos.z])

    structure_array = np.array(structure_positions, dtype=np.float32)

    if debug >= Debug.PRINT_NUMERICS:
        print(f"Structure array: {structure_array}")

    scores = []
    scores_signal_to_noise = []
    logs = []
    for j in range(10):
        start_diff_ev = time.time()

        res = optimize.differential_evolution(
            lambda params: score_fit_nonquant_array(
                structure_array,
                zmap_grid,
                # 12.0,
                1.0,
                params
            ),
            [
                # (-3, 3), (-3, 3), (-3, 3),
                # (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5),
                (-6.0, 6.0), (-6, 6.0), (-6.0, 6.0),
                (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)
            ],
            # popsize=30,
        )
        if debug >= Debug.PRINT_NUMERICS:
            print(f"Fit Score: {res.fun}")
        scores.append(res.fun)
        finish_diff_ev = time.time()
        # TODO: back to debug
        # if debug:
        # print(f"\t\t\t\tdiff ev in: {finish_diff_ev - start_diff_ev}")
        # print(f"\t\t\t\tOptimisation result: {res.x} {1-res.fun}")

        # start_basin = time.time()
        # res = optimize.basinhopping(
        #     lambda params: score_fit(
        #         probe_structure,
        #         zmap_grid,
        #         params
        #     ),
        #     x0=[0.0,0.0,0.0,0.0,0.0,0.0],
        # )
        # finish_basin = time.time()
        # if debug:
        #     print(f"\t\t\tbasin in: {finish_basin-start_basin}")
        #     print(f"\t\t\tOptimisation result: {res.x} {res.fun}")

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
            probe_structure,
            [x, y, z],
            rotation_matrix
        )

        # TODO: Remove althogether
        # optimised_structure.write_minimal_pdb(f"frag_{1-res.fun}_{str(res.x)}.pdb")

        # Score, by including the noise as well as signal
        # if debug:
        #     print(f"\t\t\t\tScoring optimized result by signal to noise")

        # score, log = score_structure_signal_to_noise_density(
        #     optimised_structure,
        #     zmap_grid,
        # )
        # score = float(res.fun) / (int(cluster.values.size) + 1)

        score, log = EXPERIMENTAL_score_structure_rscc(
            optimised_structure,
            zmap_grid,
            resolution,
            rate
        )

        scores_signal_to_noise.append(score)
        logs.append(log)
        # score = 1-float(res.fun)
        # print(f"\t\t\t\tScore: {score}")

    print(f"Best fit score: {1 - min(scores)}")
    print(f"Best signal to noise score: {max(scores_signal_to_noise)}")

    best_score_index = np.argmax(scores_signal_to_noise)
    best_score = scores_signal_to_noise[best_score_index]
    best_score_log = logs[best_score_index]
    best_score_fit_score = scores[best_score_index]

    if debug >= Debug.PRINT_NUMERICS:
        print(f"\t\t\t\tCluster size is: {int(cluster.values.size)}")
        print(f"\t\t\t\tModeled atoms % is: {float(1 - res.fun)}")
        print(f"\t\t\t\tScore is: {score}")
        # print(f"\t\t\tScoring log results are: {log}")

    return ConformerFittingResult(
        # float(best_score),
        float(
            best_score
        ),
        optimised_structure,
        {
            "fit_score": float(best_score_fit_score),
            "grid": best_score_log["grid"],
            "outer_sum": best_score_log["outer_sum"],
            "inner_sum": best_score_log["inner_sum"],
            "outer_mean": best_score_log["outer_mean"],
            "inner_mean": best_score_log["inner_mean"],
            "inner>1": best_score_log["inner>1"],
            "outer>1": best_score_log["outer>1"],
            "inner>2": best_score_log["inner>2"],
            "outer>2": best_score_log["outer>2"],
            "num_inner": best_score_log["num_inner"],
            "num_outer": best_score_log["num_outer"],
            "all_scores": best_score_log["scores"],
            "scores_from_calc": best_score_log["scores_from_calc"]
        }
    )

    # def step_func(skeleton_score):
    #     if 1 - skeleton_score > 0.40:
    #         return 1.0
    #     else:
    #         return 0.0
    #
    # def step_func(val, cut):
    #     if val > cut:
    #         return 1.0
    #     else:
    #         return 0.0
    #
    # percent_signal = best_score_log["signal"] / best_score_log["signal_samples_shape"]
    # percent_noise = best_score_log["noise"] / best_score_log["noise_samples_shape"]
    #
    # return ConformerFittingResult(
    #     # float(best_score),
    #     float(
    #         int(
    #             best_score_log["signal"]
    #             * step_func(percent_signal, 0.4)
    #             * step_func(1 - percent_noise, 0.80)
    #         )
    #         - int(best_score_log["noise"])
    #     ),
    #     optimised_structure,
    #     {
    #         "fit_score": float(best_score_fit_score),
    #         "Signal": int(best_score_log["signal"]),
    #         "Noise": int(best_score_log["noise"]),
    #         "Num Signal Samples": int(best_score_log["signal_samples_shape"]),
    #         "Num Noise Samples": int(best_score_log["noise_samples_shape"]),
    #         "Penalty": int(best_score_log["penalty"]),
    #     }
    # )


def score_conformer_array(cluster: Cluster,
                          conformer,
                          zmap_grid,
                          debug: Debug = Debug.DEFAULT) -> ConformerFittingResultInterface:
    # Center the conformer at the cluster
    centroid_cart = cluster.centroid

    if debug >= Debug.PRINT_NUMERICS:
        print(f"\t\t\t\tCartesian centroid of event is: {centroid_cart}")

    centered_structure = center_structure(
        conformer,
        centroid_cart,
    )

    # Get the probe structure
    probe_structure = get_probe_structure(centered_structure)

    if debug >= Debug.PRINT_NUMERICS:
        print(f"\t\t\t\tprobe structure: {probe_structure}")

    # Optimise
    if debug >= Debug.PRINT_NUMERICS:
        print(f"\t\t\t\tOptimizing structure fit...")

    structure_positions = []

    for model in probe_structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element.name != "H":
                        pos = atom.pos
                        structure_positions.append([pos.x, pos.y, pos.z])

    structure_array = np.array(structure_positions, dtype=np.float32)

    if debug >= Debug.PRINT_NUMERICS:
        print(f"Structure array: {structure_array}")

    scores = []
    scores_signal_to_noise = []
    logs = []
    for j in range(10):
        start_diff_ev = time.time()

        res = optimize.differential_evolution(
            lambda params: score_fit_array(
                structure_array,
                zmap_grid,
                # 12.0,
                1.0,
                params
            ),
            [
                # (-3, 3), (-3, 3), (-3, 3),
                # (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5),
                (-6.0, 6.0), (-6, 6.0), (-6.0, 6.0),
                (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)
            ],
            # popsize=30,
        )
        if debug >= Debug.PRINT_NUMERICS:
            print(f"Fit Score: {res.fun}")
        scores.append(res.fun)
        finish_diff_ev = time.time()
        # TODO: back to debug
        # if debug:
        # print(f"\t\t\t\tdiff ev in: {finish_diff_ev - start_diff_ev}")
        # print(f"\t\t\t\tOptimisation result: {res.x} {1-res.fun}")

        # start_basin = time.time()
        # res = optimize.basinhopping(
        #     lambda params: score_fit(
        #         probe_structure,
        #         zmap_grid,
        #         params
        #     ),
        #     x0=[0.0,0.0,0.0,0.0,0.0,0.0],
        # )
        # finish_basin = time.time()
        # if debug:
        #     print(f"\t\t\tbasin in: {finish_basin-start_basin}")
        #     print(f"\t\t\tOptimisation result: {res.x} {res.fun}")

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
            probe_structure,
            [x, y, z],
            rotation_matrix
        )

        # TODO: Remove althogether
        # optimised_structure.write_minimal_pdb(f"frag_{1-res.fun}_{str(res.x)}.pdb")

        # Score, by including the noise as well as signal
        # if debug:
        #     print(f"\t\t\t\tScoring optimized result by signal to noise")

        # score, log = score_structure_signal_to_noise_density(
        #     optimised_structure,
        #     zmap_grid,
        # )
        # score = float(res.fun) / (int(cluster.values.size) + 1)

        score, log = EXPERIMENTAL_score_structure_signal_to_noise_density(
            optimised_structure,
            zmap_grid,
        )

        scores_signal_to_noise.append(score)
        logs.append(log)
        # score = 1-float(res.fun)
        # print(f"\t\t\t\tScore: {score}")

    print(f"Best fit score: {1 - min(scores)}")
    print(f"Best signal to noise score: {max(scores_signal_to_noise)}")

    best_score_index = np.argmax(scores_signal_to_noise)
    best_score = scores_signal_to_noise[best_score_index]
    best_score_log = logs[best_score_index]
    best_score_fit_score = scores[best_score_index]

    if debug >= Debug.PRINT_NUMERICS:
        print(f"\t\t\t\tCluster size is: {int(cluster.values.size)}")
        print(f"\t\t\t\tModeled atoms % is: {float(1 - res.fun)}")
        print(f"\t\t\t\tScore is: {score}")
        # print(f"\t\t\tScoring log results are: {log}")

    # def step_func(skeleton_score):
    #     if 1 - skeleton_score > 0.40:
    #         return 1.0
    #     else:
    #         return 0.0

    def step_func(val, cut):
        if val > cut:
            return 1.0
        else:
            return 0.0

    percent_signal = best_score_log["signal"] / best_score_log["signal_samples_shape"]
    percent_noise = best_score_log["noise"] / best_score_log["noise_samples_shape"]

    return ConformerFittingResult(
        # float(best_score),
        float(
            int(
                best_score_log["signal"]
                * step_func(percent_signal, 0.4)
                * step_func(1 - percent_noise, 0.80)
            )
            - int(best_score_log["noise"])
        ),
        optimised_structure,
        {
            "fit_score": float(best_score_fit_score),
            "Signal": int(best_score_log["signal"]),
            "Noise": int(best_score_log["noise"]),
            "Num Signal Samples": int(best_score_log["signal_samples_shape"]),
            "Num Noise Samples": int(best_score_log["noise_samples_shape"]),
            "Penalty": int(best_score_log["penalty"]),
        }
    )


def score_conformer(cluster: Cluster, conformer, zmap_grid, debug=False):
    # Center the conformer at the cluster
    centroid_cart = cluster.centroid

    if debug:
        print(f"\t\t\t\tCartesian centroid of event is: {centroid_cart}")

    centered_structure = center_structure(
        conformer,
        centroid_cart,
    )

    # Get the probe structure
    probe_structure = get_probe_structure(centered_structure)

    if debug:
        print(f"\t\t\t\tprobe structure: {probe_structure}")

    # Optimise
    if debug:
        print(f"\t\t\t\tOptimizing structure fit...")

    # start_shgo = time.time()
    # res = optimize.shgo(
    #     lambda params: score_fit(
    #         probe_structure,
    #         zmap_grid,
    #         # 12,
    #         1.0,
    #         params
    #     ),
    #     [
    #         # (-3, 3), (-3, 3), (-3, 3),
    #         (-6.5, 6), (-6.5, 6), (-6.5, 6),
    #         # (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5),
    #         (0, 1), (0, 1), (0, 1)
    #     ],
    #     sampling_method='sobol',
    #     n=64,
    #     iters=5,
    # )
    # finish_shgo=time.time()
    # if debug:
    #     print(f"\t\t\tSHGO in: {finish_shgo-start_shgo}")
    #     print(f"\t\t\tOptimisation result: {res.x} {res.fun}")
    # print(f"\t\t\tOptimisation result: {res.xl} {res.funl}")
    start_diff_ev = time.time()

    scores = []
    # for j in range(1000):
    #
    #     distance = 12.0
    #
    #     x = (np.random.rand() - 0.5) * distance
    #     y = (np.random.rand() - 0.5)* distance
    #     z = (np.random.rand() - 0.5)* distance
    #     a = np.random.rand()
    #     b = np.random.rand()
    #     c = np.random.rand()
    #
    #     x0 = np.array([x, y, z, a ,b ,c])
    #
    #     res = scipy.optimize.minimize(
    #         lambda params: score_fit(
    #         probe_structure,
    #         zmap_grid,
    #         1.0,
    #         params,
    #         ),
    #             x0,
    #         method="nelder-mead"
    #     )
    #     scores.append(res.fun)
    #
    #     # print(f"\t\t\t\tdiff ev in: {finish_diff_ev - start_diff_ev}")
    #     print(f"\t\t\t\tOptimisation result: {res.x} {1-res.fun}")
    #
    # print(f"{1-min(scores)}")

    for j in range(10):
        res = optimize.differential_evolution(
            lambda params: score_fit(
                probe_structure,
                zmap_grid,
                # 12.0,
                1.0,
                params
            ),
            [
                # (-3, 3), (-3, 3), (-3, 3),
                # (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5),
                (-6.0, 6.0), (-6, 6.0), (-6.0, 6.0),
                (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)
            ],
            # popsize=30,
        )
        scores.append(res.fun)
        finish_diff_ev = time.time()
        # TODO: back to debug
        # if debug:
        # print(f"\t\t\t\tdiff ev in: {finish_diff_ev - start_diff_ev}")
        print(f"\t\t\t\tOptimisation result: {res.x} {1 - res.fun}")

    print(f"{1 - min(scores)}")

    # start_basin = time.time()
    # res = optimize.basinhopping(
    #     lambda params: score_fit(
    #         probe_structure,
    #         zmap_grid,
    #         params
    #     ),
    #     x0=[0.0,0.0,0.0,0.0,0.0,0.0],
    # )
    # finish_basin = time.time()
    # if debug:
    #     print(f"\t\t\tbasin in: {finish_basin-start_basin}")
    #     print(f"\t\t\tOptimisation result: {res.x} {res.fun}")

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
    rotation_matrix: np.ndarray = rotation.as_matrix()
    optimised_structure = transform_structure(
        probe_structure,
        [x, y, z],
        rotation_matrix
    )

    # TODO: Remove althogether
    # optimised_structure.write_minimal_pdb(f"frag_{1-res.fun}_{str(res.x)}.pdb")

    # Score, by including the noise as well as signal
    # if debug:
    #     print(f"\t\t\t\tScoring optimized result by signal to noise")

    # score, log = score_structure_signal_to_noise_density(
    #     optimised_structure,
    #     zmap_grid,
    # )
    # score = float(res.fun) / (int(cluster.values.size) + 1)

    score, log = EXPERIMENTAL_score_structure_signal_to_noise_density(
        optimised_structure,
        zmap_grid,
    )
    # score = 1-float(res.fun)
    print(f"\t\t\t\tScore: {score}")

    if debug:
        print(f"\t\t\t\tCluster size is: {int(cluster.values.size)}")
        print(f"\t\t\t\tModeled atoms % is: {float(1 - res.fun)}")
        print(f"\t\t\t\tScore is: {score}")
        # print(f"\t\t\tScoring log results are: {log}")

    return float(score), optimised_structure


def score_fragment_conformers(cluster, fragment_conformers: ConformersInterface, zmap_grid, res, rate,
                              debug: Debug = Debug.DEFAULT) -> LigandFittingResultInterface:
    if debug >= Debug.PRINT_NUMERICS:
        print("\t\t\t\tGetting fragment conformers from model")

    if debug >= Debug.PRINT_NUMERICS:
        print(f"\t\t\t\tScoring conformers")
    results: ConformerFittingResultsInterface = {}
    for conformer_id, conformer in fragment_conformers.conformers.items():
        # results[conformer_id] = score_conformer(cluster, conformer, zmap_grid, debug)
        # results[conformer_id] = score_conformer_array(cluster, conformer, zmap_grid, debug)
        results[conformer_id] = score_conformer_nonquant_array(cluster, conformer, zmap_grid, res, rate, debug)

    # scores = {conformer_id: result[0] for conformer_id, result in results.items()}
    # structures = {conformer_id: result[1] for conformer_id, result in results.items()}

    # if debug >= Debug.PRINT_NUMERICS:
    #     print(f"\t\t\t\tConformer scores are: {scores}")

    # highest_scoring_conformer = max(scores, key=lambda x: scores[x])
    # highest_score = scores[highest_scoring_conformer]
    # highest_scoring_structure = structures[highest_scoring_conformer]

    return LigandFittingResult(
        results,
        fragment_conformers
    )


def score_cluster(cluster, zmap_grid: gemmi.FloatGrid, fragment_conformers: ConformersInterface, res, rate,
                  debug: Debug = Debug.DEFAULT) -> EventScoringResultInterface:
    if debug:
        print(f"\t\t\t\tScoring cluster")
    ligand_fitting_result = score_fragment_conformers(cluster, fragment_conformers, zmap_grid, res, rate, debug)

    return EventScoringResult(ligand_fitting_result)


class LigandFittingResult(LigandFittingResultInterface):
    def __init__(self,
                 conformer_fitting_results: ConformerFittingResultsInterface,
                 conformers: ConformersInterface,
                 # selected_conformer: int
                 ):
        self.conformer_fitting_results: ConformerFittingResultsInterface = conformer_fitting_results
        # selected_conformer: int = selected_conformer
        self.conformers = conformers

    def log(self) -> Dict:
        return {
            "Conformer results": {
                str(conformer_id): conformer_result.log()
                for conformer_id, conformer_result
                in self.conformer_fitting_results.items()
            },
            "Conformer generation info": self.conformers.log()
        }


def score_clusters(
        clusters: Dict[Tuple[int, int], Cluster],
        zmaps,
        fragment_dataset,
        res, rate,
        debug: Debug = Debug.DEFAULT,
) -> Dict[Tuple[int, int], EventScoringResultInterface]:
    if debug >= Debug.PRINT_SUMMARIES:
        print(f"\t\t\tGetting fragment conformers...")
    fragment_conformers: ConformersInterface = get_conformers(fragment_dataset, debug=debug)

    results = {
        cluster_id: EventScoringResult(
            LigandFittingResult(
                {
                    0: ConformerFittingResult(
                        None,
                        None,
                        None
                    )
                },
                fragment_conformers
            )
        )
        for cluster_id, cluster
        in clusters.items()
    }

    if len(fragment_conformers.conformers) == 0:
        return results

    for cluster_id, cluster in clusters.items():
        if debug:
            print(f"\t\t\t\tProcessing cluster: {cluster_id}")

        zmap_grid = zmaps[cluster_id]

        results[cluster_id] = score_cluster(cluster, zmap_grid, fragment_conformers, res, rate, debug)

    return results


class EventScoringResult(EventScoringResultInterface):
    def __init__(self, ligand_fitting_result):
        # self.score: float = score
        self.ligand_fitting_result: LigandFittingResultInterface = ligand_fitting_result

    def get_selected_conformer_key(self) -> Optional[int]:

        keys_with_scores = [
                conformer_id
                for conformer_id
                in self.ligand_fitting_result.conformer_fitting_results
                if self.ligand_fitting_result.conformer_fitting_results[conformer_id].score is not None
            ]

        if len(keys_with_scores) == 0:
            return None

        return max(
            keys_with_scores,
            key=lambda key: self.ligand_fitting_result.conformer_fitting_results[key].score
        )

    def get_selected_conformer_results(self) -> Optional[ConformerFittingResultInterface]:
        key = self.get_selected_conformer_key()

        if key:
            return self.ligand_fitting_result.conformer_fitting_results[key]
        else:
            return None

    def get_selected_structure(self) -> Any:

        selected_conformer = self.get_selected_conformer_results()

        if selected_conformer:
            return selected_conformer.optimised_fit
        else:
            return None

    def get_selected_structure_score(self) -> Optional[float]:
        selected_conformer = self.get_selected_conformer_results()

        if selected_conformer:
            return selected_conformer.score
        else:
            return None

    def log(self) -> Dict:
        return {
            "Selected conformer id": str(self.get_selected_conformer_key()),
            "Selected conformer score": str(self.get_selected_structure_score()),
            "Ligand fitting log": self.ligand_fitting_result.log()
        }


# def calculate_optimal_contour(
#         dataset: DatasetInterface,
#         crystallographic_grid: CrystallographicGridInterface,
# ) -> float:
#     # Get Ligand conformer as structure
#
#     # Get volume approximation
#
#     #
#
#
#
#     ...

def get_event_map_reference_grid_quantised(
        reference_xmap_grid: CrystallographicGridInterface,
        zmap_grid: CrystallographicGridInterface,
        model: ModelInterface,
        event: EventInterface,
        reference_xmap_grid_array: NDArrayInterface,
        inner_mask_grid: CrystallographicGridInterface,
        outer_mask_grid: CrystallographicGridInterface,
        event_map_cut: float,
        below_cut_score: float,
        event_density_score: float,
        protein_score: float,
        protein_event_overlap_score: float,
        debug: Debug = Debug.DEFAULT
) -> Tuple[CrystallographicGridInterface, Dict]:
    event_map_reference_grid = gemmi.FloatGrid(*[reference_xmap_grid.nu,
                                                 reference_xmap_grid.nv,
                                                 reference_xmap_grid.nw,
                                                 ]
                                               )
    event_map_reference_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")  # xmap.xmap.spacegroup
    event_map_reference_grid.set_unit_cell(reference_xmap_grid.unit_cell)

    event_map_reference_grid_array = np.array(event_map_reference_grid,
                                              copy=False,
                                              )

    mean_array = model.mean
    event_map_reference_grid_array[:, :, :] = (reference_xmap_grid_array - (event.bdc.bdc * mean_array)) / (
            1 - event.bdc.bdc)

    # Mask the protein except around the event
    # inner_mask_int_array = grid.partitioning.inner_mask
    inner_mask_int_array = np.array(
        inner_mask_grid,
        copy=False,
        dtype=np.int8,
    )
    outer_mask_int_array = np.array(
        outer_mask_grid,
        copy=False,
        dtype=np.int8,
    )

    high_mask = np.zeros(inner_mask_int_array.shape, dtype=bool)
    high_mask[event_map_reference_grid_array >= event_map_cut] = True
    low_mask = np.zeros(inner_mask_int_array.shape, dtype=bool)
    low_mask[event_map_reference_grid_array < event_map_cut] = True

    if debug >= Debug.PRINT_NUMERICS:
        print(f"\t\t\tHigh mask points: {np.sum(high_mask)}")
        print(f"\t\t\tLow mask points: {np.sum(low_mask)}")

    # Rescale the map
    event_map_reference_grid_array[event_map_reference_grid_array < event_map_cut] = below_cut_score
    event_map_reference_grid_array[event_map_reference_grid_array >= event_map_cut] = event_density_score

    # Add high z mask
    zmap_array = np.array(zmap_grid)
    event_map_reference_grid_array[zmap_array > 2.0] = event_density_score

    # Event mask
    event_mask = np.zeros(inner_mask_int_array.shape, dtype=bool)
    event_mask[event.cluster.event_mask_indicies] = True
    inner_mask = np.zeros(inner_mask_int_array.shape, dtype=bool)
    inner_mask[np.nonzero(inner_mask_int_array)] = True
    outer_mask = np.zeros(inner_mask_int_array.shape, dtype=bool)
    outer_mask[np.nonzero(outer_mask_int_array)] = True

    # Mask the protein except at event sites with a penalty
    event_map_reference_grid_array[inner_mask & (~event_mask)] = protein_score

    # Mask the protein-event overlaps with zeros
    event_map_reference_grid_array[inner_mask & event_mask] = protein_event_overlap_score

    # Noise
    noise_points = event_map_reference_grid_array[outer_mask & high_mask & (~inner_mask)]
    num_noise_points = noise_points.size
    potential_noise_points = event_map_reference_grid_array[outer_mask & (~inner_mask)]
    num_potential_noise_points = potential_noise_points.size
    percentage_noise = num_noise_points / num_potential_noise_points

    noise = {
        'num_noise_points': int(num_noise_points),
        'num_potential_noise_points': int(num_potential_noise_points),
        'percentage_noise': float(percentage_noise)
    }

    if debug >= Debug.PRINT_SUMMARIES:
        print(f"\t\t\tNoise is: {noise}")

    if debug >= Debug.PRINT_SUMMARIES:
        print("\t\t\tScoring...")

    if debug >= Debug.PRINT_NUMERICS:
        print(f"\t\t\tEvent map for scoring: {np.mean(event_map_reference_grid_array)}; "
              f"{np.max(event_map_reference_grid_array)}; {np.min(event_map_reference_grid_array)}")
        print(f"\t\t\tEvent map for scoring: {np.sum(event_map_reference_grid_array == 1)}; "
              f"{np.sum(event_map_reference_grid_array == 0)}; {np.sum(event_map_reference_grid_array == -1)}")

    return event_map_reference_grid, noise


def get_event_map_reference_grid(
        reference_xmap_grid: CrystallographicGridInterface,
        zmap_grid: CrystallographicGridInterface,
        model: ModelInterface,
        event: EventInterface,
        reference_xmap_grid_array: NDArrayInterface,
        inner_mask_grid: CrystallographicGridInterface,
        outer_mask_grid: CrystallographicGridInterface,
        event_map_cut: float,
        below_cut_score: float,
        event_density_score: float,
        protein_score: float,
        protein_event_overlap_score: float,
        debug: Debug = Debug.DEFAULT
) -> Tuple[CrystallographicGridInterface, Dict]:
    event_map_reference_grid = gemmi.FloatGrid(*[reference_xmap_grid.nu,
                                                 reference_xmap_grid.nv,
                                                 reference_xmap_grid.nw,
                                                 ]
                                               )
    event_map_reference_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")  # xmap.xmap.spacegroup
    event_map_reference_grid.set_unit_cell(reference_xmap_grid.unit_cell)

    event_map_reference_grid_array = np.array(event_map_reference_grid,
                                              copy=False,
                                              )

    mean_array = model.mean
    event_map_reference_grid_array[:, :, :] = (reference_xmap_grid_array - (event.bdc.bdc * mean_array)) / (
            1 - event.bdc.bdc)

    # Mask the protein except around the event
    # inner_mask_int_array = grid.partitioning.inner_mask
    inner_mask_int_array = np.array(
        inner_mask_grid,
        copy=False,
        dtype=np.int8,
    )
    outer_mask_int_array = np.array(
        outer_mask_grid,
        copy=False,
        dtype=np.int8,
    )

    # high_mask = np.zeros(inner_mask_int_array.shape, dtype=bool)
    # high_mask[event_map_reference_grid_array >= event_map_cut] = True
    # low_mask = np.zeros(inner_mask_int_array.shape, dtype=bool)
    # low_mask[event_map_reference_grid_array < event_map_cut] = True

    # if debug >= Debug.PRINT_NUMERICS:
    #     print(f"\t\t\tHigh mask points: {np.sum(high_mask)}")
    #     print(f"\t\t\tLow mask points: {np.sum(low_mask)}")

    # # Rescale the map
    # event_map_reference_grid_array[event_map_reference_grid_array < event_map_cut] = below_cut_score
    # event_map_reference_grid_array[event_map_reference_grid_array >= event_map_cut] = event_density_score

    # # Add high z mask
    # zmap_array = np.array(zmap_grid)
    # event_map_reference_grid_array[zmap_array > 2.0] = event_density_score

    # Event mask
    event_mask = np.zeros(inner_mask_int_array.shape, dtype=bool)
    event_mask[event.cluster.event_mask_indicies] = True
    inner_mask = np.zeros(inner_mask_int_array.shape, dtype=bool)
    inner_mask[np.nonzero(inner_mask_int_array)] = True
    outer_mask = np.zeros(inner_mask_int_array.shape, dtype=bool)
    outer_mask[np.nonzero(outer_mask_int_array)] = True

    # Mask the protein except at event sites with a penalty
    event_map_reference_grid_array[inner_mask & (~event_mask)] = protein_score

    # Mask the protein-event overlaps with zeros
    event_map_reference_grid_array[inner_mask & event_mask] = protein_event_overlap_score

    # Noise
    # noise_points = event_map_reference_grid_array[outer_mask & high_mask & (~inner_mask)]
    # num_noise_points = noise_points.size
    # potential_noise_points = event_map_reference_grid_array[outer_mask & (~inner_mask)]
    # num_potential_noise_points = potential_noise_points.size
    # percentage_noise = num_noise_points / num_potential_noise_points

    noise = {
        'num_noise_points': 0,
        'num_potential_noise_points': 0,
        'percentage_noise': 0,
    }

    if debug >= Debug.PRINT_SUMMARIES:
        print(f"\t\t\tNoise is: {noise}")

    if debug >= Debug.PRINT_SUMMARIES:
        print("\t\t\tScoring...")

    # if debug >= Debug.PRINT_NUMERICS:
    #     print(f"\t\t\tEvent map for scoring: {np.mean(event_map_reference_grid_array)}; "
    #           f"{np.max(event_map_reference_grid_array)}; {np.min(event_map_reference_grid_array)}")
    #     print(f"\t\t\tEvent map for scoring: {np.sum(event_map_reference_grid_array == 1)}; "
    #           f"{np.sum(event_map_reference_grid_array == 0)}; {np.sum(event_map_reference_grid_array == -1)}")

    return event_map_reference_grid, noise


class GetEventScoreInbuilt(GetEventScoreInbuiltInterface):
    tag: Literal["inbuilt"] = "inbuilt"

    def __call__(self,
                 test_dtag,
                 model_number,
                 processed_dataset,
                 dataset_xmap,
                 zmap,
                 events,
                 model,
                 grid,
                 dataset_alignment,
                 max_site_distance_cutoff,
                 min_bdc, max_bdc,
                 reference,
                 res, rate,
                 structure_output_folder,
                 event_map_cut=2.0,
                 below_cut_score=0.0,
                 event_density_score=1.0,
                 protein_score=-1.0,
                 protein_event_overlap_score=0.0,
                 debug: Debug = Debug.DEFAULT,
                 ) -> EventScoringResultsInterface:
        # Get the events and their BDCs
        if debug >= Debug.PRINT_SUMMARIES:
            print("\t\tGetting events...")

        time_event_finding_start = time.time()

        time_event_finding_finish = time.time()
        if debug >= Debug.PRINT_SUMMARIES:
            print(f"\t\tTime to find events for model: {time_event_finding_finish - time_event_finding_start}")

        # Calculate the event maps
        reference_xmap_grid = dataset_xmap.xmap
        reference_xmap_grid_array = np.array(reference_xmap_grid, copy=True)

        # Mask protein
        if debug >= Debug.PRINT_SUMMARIES:
            print("\t\tMasking protein...")
        inner_mask_grid = gemmi.Int8Grid(*[grid.grid.nu, grid.grid.nv, grid.grid.nw])
        inner_mask_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        inner_mask_grid.set_unit_cell(grid.grid.unit_cell)
        for atom in reference.dataset.structure.protein_atoms():
            pos = atom.pos
            inner_mask_grid.set_points_around(pos,
                                              radius=2.0,
                                              value=1,
                                              )

        outer_mask_grid = gemmi.Int8Grid(*[grid.grid.nu, grid.grid.nv, grid.grid.nw])
        outer_mask_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        outer_mask_grid.set_unit_cell(grid.grid.unit_cell)
        for atom in reference.dataset.structure.protein_atoms():
            pos = atom.pos
            outer_mask_grid.set_points_around(pos,
                                              radius=6.0,
                                              value=1,
                                              )

        if debug >= Debug.PRINT_SUMMARIES:
            print("\t\tIterating events...")

        event_scores: EventScoringResultsInterface = {}
        noises = {}

        time_event_scoring_start = time.time()

        for event_id, event in events.events.items():

            if debug >= Debug.PRINT_SUMMARIES:
                print("\t\t\tCaclulating event maps...")

            # event_map_reference_grid, noise = get_event_map_reference_grid_quantised(
            #     reference_xmap_grid,
            #     zmap.zmap,
            #     model,
            #     event,
            #     reference_xmap_grid_array,
            #     inner_mask_grid,
            #     outer_mask_grid,
            #     event_map_cut,
            #     below_cut_score,
            #     event_density_score,
            #     protein_score,
            #     protein_event_overlap_score,
            #     debug=debug
            # )
            event_map_reference_grid, noise = get_event_map_reference_grid(
                reference_xmap_grid,
                zmap.zmap,
                model,
                event,
                reference_xmap_grid_array,
                inner_mask_grid,
                outer_mask_grid,
                event_map_cut,
                below_cut_score,
                event_density_score,
                protein_score,
                protein_event_overlap_score,
                debug=debug
            )
            noises[event_id.event_idx.event_idx] = noise

            # Score
            time_scoring_start = time.time()
            results: Dict[Tuple[int, int], EventScoringResultInterface] = score_clusters(
                {(0, 0): event.cluster},
                {(0, 0): event_map_reference_grid},
                processed_dataset,
                res, rate,
                debug=debug,
            )
            time_scoring_finish = time.time()
            if debug >= Debug.PRINT_SUMMARIES:
                print(f"\t\t\tTime to actually score all events: {time_scoring_finish - time_scoring_start}")

            # Ouptut: actually will output only one result, so only one iteration guaranteed
            for result_id, result in results.items():
                # initial_score = result[0]
                # structure = result[1]
                initial_score = result.get_selected_structure_score()
                structure = result.get_selected_structure()



                # if initial_score < 0.4:
                #     score = 0
                # else:
                #     score = initial_score / percentage_noise

                score = initial_score

                # TODO remove
                conformer_result = result.get_selected_conformer_results()
                if conformer_result:
                    ed_grid = conformer_result.score_log["grid"]
                    ccp4 = gemmi.Ccp4Map()
                    ccp4.grid = ed_grid
                    ccp4.update_ccp4_header(2, True)
                    ccp4.setup()
                    ccp4.write_ccp4_map(str(structure_output_folder / f'{model_number}_'
                                                                      f'{event_id.event_idx.event_idx}_app.ccp4'))
                    # del result.get_selected_conformer_results().score_log["grid"]

                    for conformer_id, conformer_fitting_result in \
                            result.ligand_fitting_result.conformer_fitting_results.items():
                        if conformer_fitting_result:

                            del conformer_fitting_result.score_log["grid"]

                if debug >= Debug.INTERMEDIATE_FITS:
                    if structure:
                        structure.write_minimal_pdb(
                            str(
                                structure_output_folder / f'{model_number}_{event_id.event_idx.event_idx}.pdb'
                            )
                        )

                string = f"\t\tModel {model_number} Event {event_id.event_idx.event_idx} Score {score} Event Size " \
                         f"{event.cluster.values.size}"
                print(string)

                event_scores[event_id] = result

                # EventScoringResult(
                #     score,
                #     result,
                # )

        time_event_scoring_finish = time.time()

        if debug >= Debug.PRINT_SUMMARIES:
            print(f"\t\tTime to score all events: {time_event_scoring_finish - time_event_scoring_start}. Num events: "
                  f"{len(events.events)}")

        return event_scores
