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


def get_structures_from_mol(mol: Chem.Mol, max_conformers) -> MutableMapping[int, gemmi.Structure]:
    fragmentstructures: MutableMapping[int, gemmi.Structure] = {}
    for i, conformer in enumerate(mol.GetConformers()):
        if i > max_conformers:
            continue
        else:
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

            fragmentstructures[i] = structure

    return fragmentstructures


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


def structures_from_cif(source_ligand_cif, debug=False):
    # doc = gemmi.cif.read_file(str(source_ligand_cif))
    # block = doc[-1]
    # cc = gemmi.make_chemcomp_from_block(block)
    # cc.remove_hydrogens()
    # small_structure = gemmi.read_small_structure(str(source_ligand_cif))
    cif_doc = gemmi.cif.read(str(source_ligand_cif))
    # for block in cif_doc:
    small_structure = gemmi.make_small_structure_from_block(cif_doc[-1])
    if debug:
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


def get_conformers(
        fragment_dataset,
        pruning_threshold=5,
        num_pose_samples=100,
        max_conformers=10,
        debug=False,
) -> MutableMapping[int, Chem.Mol]:
    # Decide how to load
    fragment_structures = {}
    if fragment_dataset.source_ligand_smiles:

        if debug:
            print(f'\t\tGetting mol from ligand smiles')
        mol = get_fragment_mol_from_dataset_smiles_path(fragment_dataset.source_ligand_smiles)

        # Generate conformers
        m2: Chem.Mol = Chem.AddHs(mol)

        # Generate conformers
        cids = AllChem.EmbedMultipleConfs(m2, numConfs=num_pose_samples, pruneRmsThresh=pruning_threshold)

        # Translate to structures
        fragment_structures: MutableMapping[int, gemmi.Structure] = get_structures_from_mol(m2, max_conformers)
        if len(fragment_structures) > 0:
            return fragment_structures

    if fragment_dataset.source_ligand_cif:
        if debug:
            print(f'\t\tGetting mol from cif')
        fragment_structures = structures_from_cif(fragment_dataset.source_ligand_cif, debug)
        if len(fragment_structures) > 0:
            return fragment_structures

    if fragment_dataset.source_ligand_pdb:
        if debug:
            print(f'\t\tGetting mol from ligand pdb')
        fragment_structures = {0: gemmi.read_structure(str(fragment_dataset.source_ligand_pdb))}
        if len(fragment_structures) > 0:
            return fragment_structures

    if debug:
        print(fragment_structures)

    return fragment_structures


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

    x_2 = distance*x
    y_2 = distance*y
    z_2 = distance*z

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
    res = optimize.shgo(
        lambda params: score_fit(
            probe_structure,
            zmap_grid,
            12,
            params
        ),
        [
            # (-3, 3), (-3, 3), (-3, 3),
            # (-6, 6), (-6, 6), (-6, 6),
            (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5),
            (0, 1), (0, 1), (0, 1)
        ],
        sampling_method='sobol',
        n=64,
        iters=5,
    )
    # finish_shgo=time.time()
    # if debug:
    #     print(f"\t\t\tSHGO in: {finish_shgo-start_shgo}")
    #     print(f"\t\t\tOptimisation result: {res.x} {res.fun}")
    # print(f"\t\t\tOptimisation result: {res.xl} {res.funl}")
    start_diff_ev = time.time()

    # res = optimize.differential_evolution(
    #     lambda params: score_fit(
    #         probe_structure,
    #         zmap_grid,
    #         12.0,
    #         params
    #     ),
    #     [
    #         # (-3, 3), (-3, 3), (-3, 3),
    #         (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5),
    #         #         (-6, 6), (-6, 6), (-6, 6),
    #         (0, 1), (0, 1), (0, 1)
    #     ],
    # )
    finish_diff_ev = time.time()
    # TODO: back to debug
    # if debug:
    print(f"\t\t\t\tdiff ev in: {finish_diff_ev - start_diff_ev}")
    print(f"\t\t\t\tOptimisation result: {res.x} {res.fun}")

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
    if debug:
        print(f"\t\t\t\tScoring optimized result by signal to noise")

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

    if debug:
        print(f"\t\t\t\tCluster size is: {int(cluster.values.size)}")
        print(f"\t\t\t\tModeled atoms % is: {float(1 - res.fun)}")
        print(f"\t\t\t\tScore is: {score}")
        # print(f"\t\t\tScoring log results are: {log}")

    return float(score), optimised_structure


def score_fragment_conformers(cluster, fragment_conformers, zmap_grid, debug=False):
    if debug:
        print("\t\t\t\tGetting fragment conformers from model")

    if debug:
        print(f"\t\t\t\tScoring conformers")
    results = {}
    for conformer_id, conformer in fragment_conformers.items():
        results[conformer_id] = score_conformer(cluster, conformer, zmap_grid, debug)

    scores = {conformer_id: result[0] for conformer_id, result in results.items()}
    structures = {conformer_id: result[1] for conformer_id, result in results.items()}

    if debug:
        print(f"\t\t\t\tConformer scores are: {scores}")

    highest_scoring_conformer = max(scores, key=lambda x: scores[x])
    highest_score = scores[highest_scoring_conformer]
    highest_scoring_structure = structures[highest_scoring_conformer]

    return highest_score, highest_scoring_structure


def score_cluster(cluster, zmap_grid: gemmi.FloatGrid, fragment_conformers, debug=False):
    if debug:
        print(f"\t\t\t\tScoring cluster")
    score, structure = score_fragment_conformers(cluster, fragment_conformers, zmap_grid, debug)

    return score, structure


def score_clusters(
        clusters: Dict[Tuple[int, int], Cluster],
        zmaps,
        fragment_dataset,
        debug=False,
):
    if debug:
        print(f"\t\t\tGetting fragment conformers...")
    fragment_conformers = get_conformers(fragment_dataset, debug=debug)

    results = {}

    if len(fragment_conformers) == 0:
        return results

    for cluster_id, cluster in clusters.items():
        if debug:
            print(f"\t\t\t\tProcessing cluster: {cluster_id}")

        zmap_grid = zmaps[cluster_id]

        results[cluster_id] = score_cluster(cluster, zmap_grid, fragment_conformers, debug)

    return results


class GetEventScoreInbuilt(GetEventScoreInbuiltInterface):
    tag: Literal["inbuilt"] = "inbuilt"

    def __call__(self,
                 test_dtag,
                 model_number,
                 processed_dataset,
                 dataset_xmap,
                 events,
                 model,
                 grid,
                 dataset_alignment,
                 max_site_distance_cutoff,
                 min_bdc, max_bdc,
                 reference,
                 structure_output_folder,
                 event_map_cut=2.0,
                 below_cut_score=0.0,
                 event_density_score=1.0,
                 protein_score=-1.0,
                 protein_event_overlap_score=0.0,
                 debug=True,
                 ):
        # Get the events and their BDCs
        if debug:
            print("\t\tGetting events...")

        time_event_finding_start = time.time()

        time_event_finding_finish = time.time()
        if debug:
            print(f"\t\tTime to find events for model: {time_event_finding_finish - time_event_finding_start}")

        # Calculate the event maps
        reference_xmap_grid = dataset_xmap.xmap
        reference_xmap_grid_array = np.array(reference_xmap_grid, copy=True)

        # Mask protein
        if debug:
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

        if debug:
            print("\t\tIterating events...")

        event_scores = {}
        noises = {}

        time_event_scoring_start = time.time()

        for event_id, event in events.events.items():

            if debug:
                print("\t\t\tCaclulating event maps...")
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

            # Rescale the map
            event_map_reference_grid_array[event_map_reference_grid_array < event_map_cut] = below_cut_score
            event_map_reference_grid_array[event_map_reference_grid_array >= event_map_cut] = event_density_score

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
            noises[event_id.event_idx.event_idx] = noise

            if debug:
                print(f"\t\t\tNoise is: {noise}")

            if debug:
                print("\t\t\tScoring...")

            # Score
            time_scoring_start = time.time()
            results = score_clusters(
                {(0, 0): event.cluster},
                {(0, 0): event_map_reference_grid},
                processed_dataset,
                debug=debug,
            )
            time_scoring_finish = time.time()
            if debug:
                print(f"\t\t\tTime to actually score all events: {time_scoring_finish - time_scoring_start}")

            # Ouptut
            for result_id, result in results.items():
                initial_score = result[0]
                structure = result[1]

                # if initial_score < 0.4:
                #     score = 0
                # else:
                #     score = initial_score / percentage_noise

                score = initial_score

                if debug:
                    structure.write_minimal_pdb(
                        str(
                            structure_output_folder / f'{model_number}_{event_id.event_idx.event_idx}.pdb'
                        )
                    )

                string = f"\t\tModel {model_number} Event {event_id.event_idx.event_idx} Score {score} Event Size " \
                         f"{event.cluster.values.size}"
                print(string)

                event_scores[event_id] = score

        time_event_scoring_finish = time.time()

        if debug:
            print(f"\t\tTime to score all events: {time_event_scoring_finish - time_event_scoring_start}. Num events: "
                  f"{len(events.events)}")

        return event_scores
