import dataclasses
from pathlib import Path

import fire
import yaml
import numpy as np
import pandas as pd
import gemmi
import networkx as nx
import networkx.algorithms.isomorphism as iso
import torch


from pandda_gemmi import constants
from pandda_gemmi.cnn import resnet18
from pandda_gemmi.event_model.score import get_sample_transform_from_event, sample_xmap
from pandda_gemmi.dmaps import save_dmap, load_dmap


@dataclasses.dataclass
class EventMatchingSpec:
    pandda_2_dir: Path
    known_hits_dir: Path


@dataclasses.dataclass
class LigandMatchingSpec:
    pandda_2_dir: Path
    known_hits_dir: Path


@dataclasses.dataclass
class EventRankingSpec:
    ...


@dataclasses.dataclass
class PRCalibrationSpec:
    pandda_2_dir: Path
    known_hits_dir: Path
    model_path: Path



def get_closest_symmetry_pos(
        pos,
        ligand_centroid,
        known_hit_structure
):
    # Contruct poss
    pos_1 = gemmi.Position(pos[0], pos[1], pos[2])
    pos_2 = gemmi.Position(ligand_centroid[0], ligand_centroid[1], ligand_centroid[2])

    # Get closest sym pos
    distance = known_hit_structure.cell.find_nearest_image(
        pos_1,
        pos_2,
    ).dist()

    return distance


def get_closest_event(
        event_table,
        ligand_centroid,
        known_hit_structure
):
    # Handle no events for dataset
    if len(event_table) == 0:
        return (None, None)

    # Iterate over events
    distances = []
    for idx, row in event_table.iterrows():
        pos = [row["x"], row['y'], row['z']]
        distance = get_closest_symmetry_pos(
            pos,
            ligand_centroid,
            known_hit_structure
        )
        distances.append(
            (
                distance,
                row
            )
        )

    return min(
        distances,
        key=lambda x: x[0]
    )


def read_known_hit_dir(known_hits_dir):
    structures = {}
    for st_path in known_hits_dir.glob("*"):
        structures[st_path.stem] = gemmi.read_structure(str(st_path))

    return structures


def get_known_hit_centroids(structures):
    centroids = {}
    for structure_key, structure in structures.items():
        centroids[structure_key] = {}
        for model in structure:
            for chain in model:
                for res in chain:
                    if res.name in ["LIG", "XXX"]:
                        poss = []
                        for atom in res:
                            pos = atom.pos
                            poss.append([pos.x, pos.y, pos.z])
                        centroid = np.mean(poss, axis=0)
                        centroids[structure_key][f"{chain.name}_{res.seqid.num}"] = centroid

    return centroids


def match_events(spec: EventMatchingSpec):
    # Get the event table
    event_table = pd.read_csv(spec.pandda_2_dir / constants.PANDDA_ANALYSES_DIR / constants.PANDDA_ANALYSE_EVENTS_FILE)

    # Get known hits
    known_hit_structures = read_known_hit_dir(spec.known_hits_dir)

    # Get the known hit ligand centroids
    known_hit_centroids = get_known_hit_centroids(known_hit_structures)

    # For each known hit, for each ligand, check the distance to the (symmetrically) closest event
    records = []
    for dtag, ligand_centroids in known_hit_centroids.items():
        for ligand_key, ligand_centroid in ligand_centroids.items():
            distance, event_row = get_closest_event(
                event_table[event_table["dtag"] == dtag],
                ligand_centroid,
                known_hit_structures[dtag]
            )
            records.append(
                {
                    "Dtag": dtag,
                    "Ligand Key": ligand_key,
                    "Distance": distance
                }
            )

    # Contruct the table
    df = pd.DataFrame(records)

    return df


def get_known_hits(known_hit_structures):
    centroids = {}
    for structure_key, structure in known_hit_structures.items():
        centroids[structure_key] = {}
        for model in structure:
            for chain in model:
                for res in chain:
                    if res.name in ["LIG", "XXX"]:
                        centroids[structure_key][f"{chain.name}_{res.seqid.num}"] = res

    return centroids


def get_autobuilds(pandda_2_dir):
    processed_datasets_dir = pandda_2_dir / constants.PANDDA_PROCESSED_DATASETS_DIR
    autobuild_dir = pandda_2_dir / "autobuild"
    autobuilds = {}
    for processed_dataset_dir in processed_datasets_dir.glob("*"):
        dtag = processed_dataset_dir.name
        autobuilds[dtag] = {}
        processed_dataset_yaml = processed_dataset_dir / "processed_dataset.yaml"

        if not processed_dataset_yaml.exists():
            continue

        with open(processed_dataset_yaml, 'r') as f:
            data = yaml.safe_load(f)

        selected_model = data['Summary']['Selected Model']
        # selected_model_events = data['Summary']['Selected Model Events']

        for model, model_info in data['Models'].items():
            if model == selected_model:
                selected = True
            else:
                selected = False
            for event_idx, event_info in model_info['Events'].items():
                # if event_idx not in selected_model_events:
                #     continue

                autobuild_file = event_info['Build Path']
                autobuilds[dtag][(model, event_idx,)] = {
                    "build_path": autobuild_file,
                    "build_key": event_info['Ligand Key'],
                    'Score': event_info['Score'],
                    'Size': event_info['Size'],
                    'Local Strength': event_info['Local Strength'],
                    'RSCC': event_info['RSCC'],
                    'Signal': event_info['Signal'],
                    'Noise': event_info['Noise'],
                    'Signal/Noise': event_info['Signal'] / event_info['Noise'],
                    'X_ligand': event_info['Ligand Centroid'][0],
                    'Y_ligand': event_info['Ligand Centroid'][1],
                    'Z_ligand': event_info['Ligand Centroid'][2],
                    'X': event_info['Centroid'][0],
                    'Y': event_info['Centroid'][1],
                    'Z': event_info['Centroid'][2],
                    'Selected': selected,
                    "BDC": event_info['BDC']
                }

    return autobuilds


def get_pandda_2_autobuilt_structures(autobuilds):
    autobuilt_structures = {}
    for dtag, dtag_builds in autobuilds.items():
        autobuilt_structures[dtag] = {}
        for build_key, build_info in dtag_builds.items():
            autobuilt_structures[dtag][build_key] = gemmi.read_structure(build_info["build_path"])

    return autobuilt_structures


def get_ligand_cif_graph_matches(cif_path):
    # Open the cif document with gemmi
    cif = gemmi.cif.read(str(cif_path))

    key = "comp_LIG"
    try:
        cif['comp_LIG']
    except:
        key = "data_comp_XXX"

    # Find the relevant atoms loop
    atom_id_loop = list(cif[key].find_loop('_chem_comp_atom.atom_id'))
    atom_type_loop = list(cif[key].find_loop('_chem_comp_atom.type_symbol'))
    # atom_charge_loop = list(cif[key].find_loop('_chem_comp_atom.charge'))

    # Find the bonds loop
    bond_1_id_loop = list(cif[key].find_loop('_chem_comp_bond.atom_id_1'))
    bond_2_id_loop = list(cif[key].find_loop('_chem_comp_bond.atom_id_2'))
    bond_type_loop = list(cif[key].find_loop('_chem_comp_bond.type'))
    aromatic_bond_loop = list(cif[key].find_loop('_chem_comp_bond.aromatic'))

    # Construct the graph nodes
    G = nx.Graph()

    for atom_id, atom_type in zip(atom_id_loop, atom_type_loop):
        if atom_type == "H":
            continue
        G.add_node(atom_id, Z=atom_type)

    # Construct the graph edges
    for atom_id_1, atom_id_2 in zip(bond_1_id_loop, bond_2_id_loop):
        if atom_id_1 not in G:
            continue
        if atom_id_2 not in G:
            continue
        G.add_edge(atom_id_1, atom_id_2)

    # Get the isomorphisms
    GM = iso.GraphMatcher(G, G, node_match=iso.categorical_node_match('Z', 0))

    return [x for x in GM.isomorphisms_iter()]


def get_ligand_graphs(autobuilds, pandda_2_dir):
    ligand_graphs = {}
    for dtag, dtag_builds in autobuilds.items():
        ligand_graphs[dtag] = {}
        for build_key, build_info in dtag_builds.items():
            ligand_key = build_info["build_key"]
            if ligand_key not in ligand_graphs[dtag]:
                ligand_graphs[dtag][ligand_key] = get_ligand_cif_graph_matches(
                    pandda_2_dir / constants.PANDDA_PROCESSED_DATASETS_DIR / dtag / constants.PANDDA_LIGAND_FILES_DIR / f"{ligand_key}.cif"
                )

    return ligand_graphs


def get_rmsd(
        known_hit,
        autobuilt_structure,
        known_hit_structure,
        ligand_graph
):
    # Iterate over each isorhpism, then get symmetric distance to the relevant atom
    iso_distances = []
    for isomorphism in ligand_graph:
        # print(isomorphism)
        distances = []
        for atom in known_hit:
            if atom.element.name == "H":
                continue
            model = autobuilt_structure[0]
            chain = model[0]
            res = chain[0]
            try:
                autobuilt_atom = res[isomorphism[atom.name]][0]
            except:
                return None
            sym_clostst_dist = known_hit_structure.cell.find_nearest_image(
                atom.pos,
                autobuilt_atom.pos,
            ).dist()
            distances.append(sym_clostst_dist)
        # print(distances)
        rmsd = np.sqrt(np.mean(np.square(distances)))
        iso_distances.append(rmsd)
    return min(iso_distances)


def match_ligands(spec: LigandMatchingSpec):
    # Get the known hits structures
    known_hit_structures = read_known_hit_dir(spec.known_hits_dir)
    print(f"Got {len(known_hit_structures)} known hit structures")

    # Get the known hits
    known_hits = get_known_hits(known_hit_structures)
    print(f"Got {len(known_hits)} known hits")

    # Get the autobuild structures and their corresponding event info
    autobuilds = get_autobuilds(spec.pandda_2_dir)
    print(f"Got {len(autobuilds)} autobuilds")
    autobuilt_structures = get_pandda_2_autobuilt_structures(autobuilds)
    print(f"Got {len(autobuilt_structures)} autobuilt structures")

    # Get the corresponding cif files
    ligand_graph_matches = get_ligand_graphs(autobuilds, spec.pandda_2_dir)
    print(f"Got {len(ligand_graph_matches)} ligand graph matches")

    # For each known hit, for each selected autobuild, graph match and symmtery match and get RMSDs
    records = []
    for dtag, dtag_known_hits in known_hits.items():
        print(dtag)
        ligand_graphs = ligand_graph_matches[dtag]
        print(f'\tGot {len(dtag_known_hits)} known hits for dtag')
        dtag_autobuilt_structures = autobuilt_structures[dtag]
        print(f"\tGot {len(dtag_autobuilt_structures)} autobuilt structures for dtag ligand")
        dtag_autobuilds = autobuilds[dtag]
        print(f"\tGot {len(dtag_autobuilds)} autobuilds for dtag ligand")

        for known_hit_key, known_hit in dtag_known_hits.items():
            # # Get the autobuilds for the dataset
            for autobuild_key, autobuilt_structure in dtag_autobuilt_structures.items():
                autobuild = dtag_autobuilds[autobuild_key]
                for ligand_key, ligand_graph_automorphisms in ligand_graphs.items():
                    # # Get the RMSD
                    rmsd = get_rmsd(
                        known_hit,
                        autobuilt_structure,
                        known_hit_structures[dtag],
                        ligand_graph_automorphisms
                    )
                    records.append(
                        {
                            "Dtag": dtag,
                            "Model IDX": autobuild_key[0],
                            "Event IDX": autobuild_key[1],
                            "Known Hit Key": known_hit_key,
                            # "Autobuild Key": autobuild_key[1],
                            "Ligand Key": ligand_key,
                            "RMSD": rmsd,
                            'Score': autobuild['Score'],
                            'Size': autobuild['Size'],
                            'Local Strength': autobuild['Local Strength'],
                            'RSCC': autobuild['RSCC'],
                            'Signal': autobuild['Signal'],
                            'Noise': autobuild['Noise'],
                            'Signal/Noise': autobuild['Signal/Noise'],
                            'X_ligand': autobuild['X_ligand'],
                            'Y_ligand': autobuild['Y_ligand'],
                            'Z_ligand': autobuild['Z_ligand'],
                            'X': autobuild['X'],
                            'Y': autobuild['Y'],
                            'Z': autobuild['Z']
                        }
                    )
    print(f"Got {len(records)} rmsds")

    # Get the table of rmsds
    df = pd.DataFrame(records)

    return df


def rank_events(spec: EventRankingSpec):
    # Get the event table
    event_table = pd.read_csv(spec.pandda_2_dir / constants.PANDDA_ANALYSES_DIR / constants.PANDDA_ANALYSE_EVENTS_FILE)

    # Get known hits
    known_hit_structures = read_known_hit_dir(spec.known_hits_dir)

    # Get PanDDA 2 known hits and update
    known_hit_structures.update(read_known_hit_dir(spec.pandda_2_known_hits_dir))

    # Get the known hit ligand centroids
    known_hit_centroids = get_known_hit_centroids(known_hit_structures)

    # For each known hit, for each ligand, check the distance to the (symmetrically) closest event
    records = []
    for idx, row in event_table.iterrows():
        dtag = row['dtag']
        ligand_centroids = known_hit_centroids[dtag]
        ligand_distances = {}
        for ligand_key, ligand_centroid in ligand_centroids.items():
            distance, event_row = get_closest_event(
                event_table[event_table["dtag"] == dtag],
                ligand_centroid,
                known_hit_structures[dtag]
            )
            ligand_distances[ligand_key] = {
                'Distance': distance,
                'Event Row': event_row
            }
        closest_ligand_key = min(ligand_distances, key=lambda _key: ligand_key[_key]['Distance'])
        closest_ligand_distance = ligand_distances[closest_ligand_key]

        records.append(
            {
                "Dtag": dtag,
                "Event IDX": row['event_idx'],
                "Ligand Key": closest_ligand_key,
                "Distance": closest_ligand_distance['Distance']
            }
        )

    # Contruct the table
    df = pd.DataFrame(records)

    return df


def perform_tests(
        test,
        test_specs,
        output_dir
):
    # Get the result tables in parallel
    results = {system: test(test_spec) for system, test_spec in test_specs.items()}

    # Collate result tables
    collated_table = pd.concat(
        results.values(),
        axis=0,
        ignore_index=True,
        keys=results.keys()
    )

    # Output
    collated_table.to_csv(
        output_dir / f"{test.__name__}.csv"
    )

def get_centroid(st):
    poss = []

    for model in st:
        for chain in model:
            for res in chain:
                if res.name in ["LIG", "XXX"]:
                    for atom in res:
                        pos = atom.pos
                        poss.append([pos.x, pos.y, pos.z])
    centroid = np.mean(poss, axis=0)

    return centroid

def get_masked_dmap(dmap, st):
    mask = gemmi.Int8Grid(dmap.nu, dmap.nv, dmap.nw)
    mask.spacegroup = gemmi.find_spacegroup_by_name("P1")
    mask.set_unit_cell(dmap.unit_cell)

    # Get the mask
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    pos = atom.pos
                    mask.set_points_around(
                        pos,
                        radius=2.5,
                        value=1,
                    )

    # Get the mask array
    mask_array = np.array(mask, copy=False)

    # Get the dmap array
    dmap_array = np.array(dmap, copy=False)

    # Mask the dmap array
    dmap_array[mask_array == 0] = 0.0

    return dmap

def score_build(autobuilt_structure, event_map,
                z_map,
                raw_xmap,
                model, dev):
    # Get the masked event map
    masked_event_map = get_masked_dmap(event_map, autobuilt_structure)

    # Get the ligand centroid
    centroid = get_centroid(autobuilt_structure)

    sample_transform = get_sample_transform_from_event(
        centroid,
        0.5,
        30
    )

    # Get the sample around the ligand centroid
    sample_array = np.zeros((30,30,30),dtype=np.float32)

    # Event map sample
    dmap_sample = sample_xmap(masked_event_map, sample_transform, sample_array)
    dmap_mean = np.mean(dmap_sample)
    dmap_std = np.std(dmap_sample)
    if np.abs(dmap_std) < 0.0000001:
        image_dmap = np.copy(sample_array)
    else:
        image_dmap = (dmap_sample[np.newaxis, :] - dmap_mean) / dmap_std

    image_dmap = (dmap_sample[np.newaxis, :] - dmap_mean) / dmap_std


    # Zmap sample
    masked_z_map = get_masked_dmap(z_map, autobuilt_structure)
    zmap_sample = sample_xmap(masked_z_map, sample_transform, sample_array)
    zmap_mean = np.mean(zmap_sample)
    zmap_std = np.std(zmap_sample)
    if np.abs(zmap_std) < 0.0000001:
        image_zmap = np.copy(sample_array)
    else:
        image_zmap = (zmap_sample[np.newaxis, :] - zmap_mean) / zmap_std

    # Raw xmap sample
    masked_xmap = get_masked_dmap(raw_xmap, autobuilt_structure)
    xmap_sample = sample_xmap(masked_xmap, sample_transform, sample_array)
    xmap_mean = np.mean(xmap_sample)
    xmap_std = np.std(xmap_sample)
    if np.abs(xmap_std) < 0.0000001:
        image_raw_xmap = np.copy(sample_array)
    else:
        image_raw_xmap = (xmap_sample[np.newaxis, :] - xmap_mean) / xmap_std

    image = np.stack([image_dmap, image_zmap, image_raw_xmap], axis=1)

    # Score the sample
    # Transfer to tensor
    image_t = torch.from_numpy(image)

    # Move tensors to device
    image_c = image_t.to(dev)

    model_annotation = model(image_c)

    # Track score
    model_annotations = model_annotation.to(torch.device("cpu")).detach().numpy()
    max_score_index = np.argmax([annotation for annotation in model_annotations[:, 1]])
    score = float(model_annotations[max_score_index, 1])

    return score



def load_model(model_path):
    if torch.cuda.is_available():
        dev = 'cuda:0'
    else:
        dev='cpu'
    cnn = resnet18(num_classes=2, num_input=3)
    # cnn_path = Path(os.path.dirname(inspect.getfile(resnet))) / "model.pt"
    cnn.load_state_dict(torch.load(model_path, map_location=dev))
    cnn.to(dev)
    cnn.eval()
    return cnn.float(), dev

def get_autobuild_event_map(
                        dataset_map,
                        mean_map,
                        bdc
                    ):

    dataset_map_array = np.array(dataset_map, copy=False)
    mean_map_array = np.array(mean_map, copy=False)

    calc_event_map_array = (dataset_map_array - (bdc * mean_map_array)) / (1-bdc)

    event_map = gemmi.FloatGrid(
        dataset_map.nu, dataset_map.nv, dataset_map.nw
    )
    event_map_array = np.array(event_map, copy=False)
    event_map_array[:, :, :] = calc_event_map_array[:, :, :]
    event_map.set_unit_cell(dataset_map.unit_cell)

    return event_map

STRUCTURE_FACTORS = (
    ('pdbx_FWT', 'pdbx_PHWT'),
    ("FWT", "PHWT"),
    ("2FOFCWT", "PH2FOFCWT"),
    ("2FOFCWT_iso-fill", "PH2FOFCWT_iso-fill"),
    ("2FOFCWT_fill", "PH2FOFCWT_fill",),
    ("2FOFCWT", "PHI2FOFCWT"),
)

def load_xmap_from_mtz(path):
    mtz = gemmi.read_mtz_file(str(path))
    for f, phi in STRUCTURE_FACTORS:
        try:
            xmap = mtz.transform_f_phi_to_map(f, phi, sample_rate=3)
            return xmap
        except Exception as e:
            continue
    raise Exception()

def calibrate_pr(spec: PRCalibrationSpec):
    # Load the model
    model, dev = load_model(spec.model_path)

    # Get the known hits structures
    known_hit_structures = read_known_hit_dir(spec.known_hits_dir)
    print(f"Got {len(known_hit_structures)} known hit structures")

    # Get the known hits
    known_hits = get_known_hits(known_hit_structures)
    print(f"Got {len(known_hits)} known hits")

    # Get the autobuild structures and their corresponding event info
    autobuilds = get_autobuilds(spec.pandda_2_dir)
    print(f"Got {len(autobuilds)} autobuilds")
    autobuilt_structures = get_pandda_2_autobuilt_structures(autobuilds)
    print(f"Got {len(autobuilt_structures)} autobuilt structures")

    # Get the corresponding cif files
    ligand_graph_matches = get_ligand_graphs(autobuilds, spec.pandda_2_dir)
    print(f"Got {len(ligand_graph_matches)} ligand graph matches")

    # For each known hit, for each selected autobuild, graph match and symmtery match and get RMSDs
    records = []
    for dtag, dtag_known_hits in known_hits.items():
        print(dtag)
        ligand_graphs = ligand_graph_matches[dtag]
        print(f'\tGot {len(dtag_known_hits)} known hits for dtag')
        dtag_autobuilt_structures = autobuilt_structures[dtag]
        print(f"\tGot {len(dtag_autobuilt_structures)} autobuilt structures for dtag ligand")
        dtag_autobuilds = autobuilds[dtag]
        print(f"\tGot {len(dtag_autobuilds)} autobuilds for dtag ligand")

        processed_dataset_dir = Path(spec.pandda_2_dir) / constants.PANDDA_PROCESSED_DATASETS_DIR
        dataset_dir = processed_dataset_dir / dtag
        dataset_map = dataset_dir / "xmap.ccp4"
        mean_map_path = dataset_dir / constants.PANDDA_MEAN_MAP_FILE.format(dtag=dtag)

        # Get the mean map
        mean_map = load_dmap(mean_map_path)

        # Get the zmap
        z_map_path = dataset_dir / constants.PANDDA_Z_MAP_FILE.format(dtag=dtag)
        z_map = load_dmap(z_map_path)

        # Get the raw xmap
        mtz_path = dataset_dir / constants.PANDDA_MTZ_FILE.format(dtag)
        raw_xmap = load_xmap_from_mtz(mtz_path)

        # Get the xmap
        dataset_map = load_dmap(dataset_map)

        for known_hit_key, known_hit in dtag_known_hits.items():
            # # Get the autobuilds for the dataset
            for autobuild_key, autobuilt_structure in dtag_autobuilt_structures.items():
                autobuild = dtag_autobuilds[autobuild_key]
                if not autobuild['Selected']:
                    continue
                # Get the BDC
                bdc = autobuild['BDC']
                for ligand_key, ligand_graph_automorphisms in ligand_graphs.items():
                    # # Get the RMSD
                    rmsd = get_rmsd(
                        known_hit,
                        autobuilt_structure,
                        known_hit_structures[dtag],
                        ligand_graph_automorphisms
                    )
                    event_map = get_autobuild_event_map(
                        dataset_map,
                        mean_map,
                        bdc
                    )
                    score = score_build(
                        autobuilt_structure,
                        event_map,
                        z_map,
                        raw_xmap,
                        model,
                        dev
                    )
                    records.append(
                        {
                            "Dtag": dtag,
                            "Model IDX": autobuild_key[0],
                            "Event IDX": autobuild_key[1],
                            "Known Hit Key": known_hit_key,
                            "Ligand Key": ligand_key,
                            "RMSD": rmsd,
                            'New Score': score
                        }
                    )
    print(f"Got {len(records)} rmsds")

    # Get the table of rmsds
    df = pd.DataFrame(records)

    return df


    ...

def run_all_tests(test_spec_yaml_path):
    # Encode the test specification
    with open(test_spec_yaml_path, 'r') as f:
        tests_spec = yaml.safe_load(f)
    match_events_old_test_specs = {
        system: EventMatchingSpec(
            Path(tests_spec[system]['PanDDA 2 Dir']),
            Path(tests_spec[system]['Known Hits Dir']),
        )
        for system
        in tests_spec
    }
    match_ligands_old_test_specs = {
        system: LigandMatchingSpec(
            Path(tests_spec[system]['PanDDA 2 Dir']),
            Path(tests_spec[system]['Known Hits Dir']),
        )
        for system
        in tests_spec
    }
    # event_ranking_test_specs = {
    #     system: EventRankingSpec(
    #         Path(tests_spec[system]['PanDDA 2 Dir']),
    #         Path(tests_spec[system]['Known Hits Dir']),
    #         Path(tests_spec[system]['PanDDA 2 Hits Dir']),
    #
    #     )
    #     for system
    #     in tests_spec
    # }
    calibrate_pr_test_specs = {
        system: PRCalibrationSpec(
            Path(tests_spec[system]['PanDDA 2 Dir']),
            Path(tests_spec[system]['Known Hits Dir']),
            Path('/dls/science/groups/i04-1/conor_dev/edanalyzer/workspace_all_data_9/all_data_ligand2.pt')
        )
        for system
        in tests_spec
    }

    # Setup output directorries
    output_dir = Path('./test_output')

    # Perform tests, collate and output

    # # Event matching, old
    # perform_tests(
    #     match_events,
    #     match_events_old_test_specs,
    #     output_dir
    # )

    # # Event matching, known new
    # perform_tests(
    #     rank_events,
    #     event_ranking_test_specs,
    #     output_dir
    # )

    # # RMSD matching, old
    # perform_tests(
    #     match_ligands,
    #     match_ligands_old_test_specs,
    #     output_dir
    # )

    # # RMSD Matching, new

    # # Event ranking, old

    # # Event ranking, new

    # # Score PR calibration
    perform_tests(
        calibrate_pr,
        calibrate_pr_test_specs,
        output_dir
    )
    ...


if __name__ == "__main__":
    fire.Fire(run_all_tests)
