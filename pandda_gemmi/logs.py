from __future__ import annotations

import json

from pprint import PrettyPrinter

printer = PrettyPrinter(indent=1)

from pandda_gemmi.pandda_types import *
from pandda_gemmi.python_types import *


def summarise_grid(grid: gemmi.FloatGrid):
    grid_array = np.array(grid, copy=False)

    summary = {
        f"Grid size": f"{grid.nu} {grid.nv} {grid.nw}",
        f"Grid spacegroup": f"{grid.spacegroup}",
        f"Grid unit cell": f"{grid.unit_cell}",
        f"Grid max": f"{np.max(grid_array)}",
        f"Grid min": f"{np.min(grid_array)}",
        f"Grid mean": f"{np.mean(grid_array)}",
    }

    return summary


def summarise_mtz(mtz: gemmi.Mtz):
    mtz_array = np.array(mtz, copy=False)

    summary = {

        f"Mtz shape": f"{mtz_array.shape}",
        f"Mtz spacegroup": f"{mtz.spacegroup}"
    }

    return summary


def summarise_structure(structure: gemmi.Structure):
    num_models: int = 0
    num_chains: int = 0
    num_residues: int = 0
    num_atoms: int = 0

    for model in structure:
        num_models += 1
        for chain in model:
            num_chains += 1
            for residue in chain:
                num_residues += 1
                for atom in residue:
                    num_atoms += 1

    summary = {
        f"Num models": f"{num_models}",
        f"Num chains": f"{num_chains}",
        f"Num residues": f"{num_residues}",
        f"Num atoms": f"{num_atoms}",
    }

    return summary


def summarise_event(event: Event):
    summary = {"Event system": f"{event.system}",
               f"Event dtag": "{event.dtag}",
               f"Event xyz": "{event.x} {event.y} {event.z}", }

    return summary


def summarise_array(array):
    summary = {
        "Shape": array.shape,
        "Mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }
    return summary


def summarise_datasets(datasets: Dict[Dtag, Dataset], pandda_fs_model: PanDDAFSModel):
    summaries = {}
    for dtag in datasets:
        dataset = datasets[dtag]
        source_dir = pandda_fs_model.data_dirs.dataset_dirs[dtag]
        processed_dir = pandda_fs_model.processed_datasets[dtag]
        if processed_dir.source_ligand_dir:
            compound_dir = processed_dir.source_ligand_dir.path
        else:
            compound_dir = None

        summary = {
            "compound_dir": str(compound_dir),
            "source_structure_path": str(processed_dir.source_pdb),
            "source_reflections_path": str(processed_dir.source_mtz),
            "ligand_cif_path": str(processed_dir.source_ligand_cif),
            "ligand_smiles_path": str(processed_dir.source_ligand_pdb),
            "ligand_pdb_path": str(processed_dir.source_ligand_pdb),
            "smoothing_factor": dataset.smoothing_factor,
            "source_dir": str(processed_dir.path),
            "processed_dir": str(source_dir.path)
        }
        summaries[dtag.dtag] = summary
    return summaries


def save_json_log(log_dict: Dict, path: Path):
    with open(str(path), "w") as f:
        json.dump(log_dict,
                  f,
                  )
