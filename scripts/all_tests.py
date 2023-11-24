import dataclasses
from pathlib import Path

import fire
import yaml
import numpy as np
import pandas as pd
import gemmi

from pandda_gemmi import constants


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
    ...


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
        with open(processed_dataset_yaml, 'r') as f:
            data = yaml.safe_load(f)

        selected_model = data['Selected Model']
        selected_model_events = data['Selected Model Events']

        for model, model_info in data['Models'].items():
            if model != selected_model:
                continue
            for event_idx, event_info in model_info['Events'].items():
                if event_idx not in selected_model_events:
                    continue

                autobuild_file = event_info['Build Path']
                autobuilds[dtag][(model, event_idx, )] = autobuild_file

    return autobuilds


def get_pandda_2_autobuilt_structures(autobuilds):
    autobuilt_structures = {}
    for dtag, dtag_builds in autobuilds.items():

        autobuilt_structures[dtag] = {}
        for build_key, build_path in dtag_builds.items():
            autobuilt_structures[build_key] = gemmi.read_structure(build_path)

    return autobuilt_structures

def get_ligand_graphs(autobuilds):

def match_ligands(spec: LigandMatchingSpec):
    # Get the known hits structures
    known_hit_structures = read_known_hit_dir(spec.known_hits_dir)

    # Get the known hits
    known_hits = get_known_hits(known_hit_structures)

    # Get the autobuild structures and their corresponding event info
    autobuilds = get_autobuilds(spec.pandda_2_dir)
    autobuilt_structures = get_pandda_2_autobuilt_structures(autobuilds)

    # Get the corresponding cif files
    ligand_graphs = get_ligand_graphs(autobuilds)

    # For each known hit, for each selected autobuild, graph match and symmtery match and get RMSDs
    records = []
    for dtag, dtag_known_hits in known_hits.items():
        ligand_graph = ligand_graphs[dtag]
        for ligand_key, known_hit in dtag_known_hits.items():
            # # Get the autobuilds for the dataset
            dtag_autobuilds = autobuilt_structures[dtag]

            for autobuild_key, autobuilt_structure in dtag_autobuilds.items():
                # # Get the RMSD
                rmsd = get_rmsd(
                    known_hit,
                    autobuilt_structure,
                    known_hit_structures[dtag],
                    ligand_graph
                )
                records.append(
                    {
                        "Dtag": dtag,
                        "Ligand Key": ligand_key,
                        "Autobuild Key": autobuild_key,
                        "RMSD": rmsd
                    }
                )

    # Get the table of rmsds
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

    # Setup output directorries
    output_dir = Path('./test_output')

    # Perform tests, collate and output

    # # Event matching, old
    perform_tests(
        match_events,
        match_events_old_test_specs,
        output_dir
    )

    # # Event matching, known new
    ...

    # # RMSD matching, old
    perform_tests(
        match_ligands,
        match_ligands_old_test_specs,
        output_dir
    )

    # # RMSD Matching, new

    # # Event ranking, old

    # # Event ranking, new

    # # Score PR calibration

    ...


if __name__ == "__main__":
    fire.Fire(run_all_tests)
