import os
import re
from pathlib import Path

import gemmi
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import fire

from pandda_gemmi import constants


def _save(std,
          ligand_samples,
          masked_difference_map,
          output_dir):
    dtype = np.dtype(
        [
            ('robust_std', 'f4',),
            ('ligand_samples', 'f4', (ligand_samples.size,)),
            ('masked_difference_map', 'f4', (masked_difference_map.size,))
        ]
    )
    print(masked_difference_map)
    print(ligand_samples)
    print(std)

    arr = np.array([(std, ligand_samples, masked_difference_map)], dtype=dtype)
    with open(output_dir, 'wb') as f:
        np.save(f, arr)
    ...


def _plot_difference_magnitude_distribution(masked_xmap):
    ...


def _robust_std_estimate(masked_xmap):
    rng = np.random.default_rng()
    normal = rng.standard_normal(size=masked_xmap.size)
    masked_xmap_sorted = np.sort(masked_xmap)
    normal_sorted = np.sort(normal)
    lower = int(normal.size / 6)
    upper = normal.size - lower
    std, mean = np.polyfit(normal_sorted[lower:upper], masked_xmap_sorted[lower:upper], deg=1)

    return std


def _get_ligand_samples(xmap, st):
    samples = []
    for model in st:
        for chain in model:
            for res in chain:
                if res.name in ['LIG', 'XXX']:
                    for atom in res:
                        if atom.element.name == 'H':
                            continue
                        samples.append(
                            xmap.interpolate_value(atom.pos)
                        )

    return np.array(samples)


def _get_delta_map(xmap_1, xmap_2):
    delta_map = gemmi.FloatGrid(xmap_1.nu, xmap_1.nv, xmap_1.nw)
    delta_map.set_unit_cell(xmap_1.unit_cell)
    delta_map_arr = np.array(delta_map, copy=False)
    arr_1 = np.array(xmap_1, copy=False)
    arr_2 = np.array(xmap_2, copy=False)

    delta_map_arr[:, :, :] = arr_1[:, :, :] - arr_2[:, :, :]

    return delta_map


def is_protein_residue(residue):
    for atom in residue:
        if "CA" in atom.name.upper():
            return True

    return False


def _mask_map_around_protein(xmap, st):
    outer_mask = gemmi.Int8Grid(xmap.nu, xmap.nv, xmap.nw)
    outer_mask.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    outer_mask.set_unit_cell(xmap.unit_cell)
    for model in st:
        for chain in model:
            for residue in chain.get_polymer().first_conformer():

                if not is_protein_residue(residue):
                    continue

                for atom in residue:
                    if atom.element.name == 'H':
                        continue
                    pos = atom.pos
                    outer_mask.set_points_around(
                        pos,
                        radius=6.0,
                        value=1,
                    )
    xmap_arr = np.array(xmap)
    outer_mask_array = np.array(outer_mask, copy=False, dtype=np.int8)
    return xmap_arr[outer_mask_array > 0]


def _get_model(dataset_dir):
    path = dataset_dir / constants.PANDDA_MODELLED_STRUCTURES_DIR / constants.PANDDA_EVENT_MODEL.format(
        dataset_dir.name)
    st = gemmi.read_structure(str(path))
    return st


def _get_pandda_mean_map(dataset_dir):
    model_maps = dataset_dir / 'model_maps'
    map_paths = {}
    for model_map in model_maps.glob("*_mean.ccp4"):
        match = re.match("([0-9]+)_mean.ccp4", model_map.name, )
        map_paths[match[0]] = model_map
    selected_map_path = map_paths[max(map_paths)]
    m = gemmi.read_ccp4_map(str(selected_map_path))
    grid = m.grid
    return grid


def _get_xmap(dataset_dir):
    m = gemmi.read_ccp4_map(str(dataset_dir / 'xmap.ccp4'))
    grid = m.grid
    return grid


def analyse_difference_distribution(dataset_dir, output_dir):
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    if not output_dir.exists():
        os.mkdir(output_dir)

    # Get the relevant mean map
    mean_map = _get_pandda_mean_map(dataset_dir)

    # Get the relveant xmap
    xmap = _get_xmap(dataset_dir)

    # Get the model
    st = _get_model(dataset_dir)

    # Get the difference map
    difference_map = _get_delta_map(xmap, mean_map)

    # Mask the difference map
    masked_difference_map = _mask_map_around_protein(difference_map, st)

    # Get the robust std estimation
    std = _robust_std_estimate(masked_difference_map)

    # Get ligand samples
    ligand_samples = _get_ligand_samples(difference_map, st)

    # Plot difference distribution
    _save(
        std,
        ligand_samples,
        masked_difference_map,
        output_dir / f'{dataset_dir.name}_difference_distributions.npz'
    )

    # Plot ligand sample distribution

    ...


if __name__ == "__main__":
    fire.Fire(analyse_difference_distribution)
