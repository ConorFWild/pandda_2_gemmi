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
class RMSDMatchingSpec:
    ...


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
    print(pos)
    print(ligand_centroid)
    pos_1 = gemmi.Position(pos[0], pos[1], pos[2])
    pos_2 = gemmi.Position(ligand_centroid[0], ligand_centroid[1], ligand_centroid[2])

    # Get closest sym pos
    distance = known_hit_structure.cell.find_nearest_image(
        pos_1,
        pos_2,
    ).dist

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

    # Setup output directorries
    output_dir = ...

    # Perform tests, collate and output

    # # Event matching, old
    perform_tests(
        match_events,
        match_events_old_test_specs,
        output_dir
    )

    # # Event matching, known new

    # # RMSD matching, old

    # # RMSD Matching, new

    # # Event ranking, old

    # # Event ranking, new

    # # Score PR calibration

    ...


if __name__ == "__main__":
    fire.Fire(run_all_tests)
