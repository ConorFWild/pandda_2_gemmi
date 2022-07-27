from __future__ import annotations

import math

import os
import dataclasses
import subprocess
from pathlib import Path
import json

from typing import *

import fire
import numpy as np
import gemmi
# import ray

from pandda_gemmi.analyse_interface import *
from pandda_gemmi import constants



def get_loci(_structure):
    loci = []
    for model in _structure:
        for chain in model:
            for residue in chain:
                for atom_1 in residue:
                    if atom_1.element.name == "H":
                        continue
                    pos_1 = atom_1.pos

                    for atom_2 in residue:
                        if atom_2.element.name == "H":
                            continue
                        pos_2 = atom_2.pos
                        if pos_1.dist(pos_2) < 2.0:
                            new_pos = gemmi.Position(
                                (pos_1.x + pos_2.x) / 2,
                                (pos_1.y + pos_2.y) / 2,
                                (pos_1.z + pos_2.z) / 2,

                            )
                            loci.append(new_pos)

    return loci


def get_samples(positions,
                sample_density=10,
                buffer=2.0,
                ):
    positions_array_list = []

    for position in positions:
        # get positions in an arrart
        position_array = np.array([position.x, position.y, position.z]).reshape((1, 3))
        positions_array_list.append(position_array)

    positions_array = np.vstack(positions_array_list)

    min_pos = np.min(positions_array, axis=0) - buffer
    max_pos = np.max(positions_array, axis=0) + buffer

    xs = np.linspace(min_pos[0], max_pos[0], num=int((max_pos[0] - min_pos[0]) * sample_density))
    ys = np.linspace(min_pos[1], max_pos[1], num=int((max_pos[1] - min_pos[1]) * sample_density))
    zs = np.linspace(min_pos[2], max_pos[2], num=int((max_pos[2] - min_pos[2]) * sample_density))

    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs)

    flat_grid_x = grid_x.flatten()
    flat_grid_y = grid_y.flatten()
    flat_grid_z = grid_z.flatten()

    samples_array = np.hstack(
        [
            flat_grid_x.reshape((len(flat_grid_x), 1)),
            flat_grid_y.reshape((len(flat_grid_y), 1)),
            flat_grid_z.reshape((len(flat_grid_z), 1)),
        ])

    return positions_array, samples_array


def get_sample_distances(
        positions_array,
        samples_array,
):
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(positions_array)
    distances, indices = nbrs.kneighbors(samples_array)

    return distances
    # valid_samples = samples_array[(distances > 0.95 * radius).flatten(), :]


def truncate_samples(samples, distances, r_0, r_1):
    r_0_mask = (distances > r_0).flatten()
    r_1_mask = (distances <= r_1).flatten()
    mask = r_0_mask * r_1_mask
    valid_samples = samples[mask, :]
    return valid_samples



def EXPERIMENTAL_noise_from_samples(noise_samples, xmap, cutoff):
    noise_log = {}
    samples_are_noise = []
    for sample in noise_samples:
        pos = gemmi.Position(*sample)

        value = xmap.interpolate_value(pos)

        # Check if they are over cutoff
        if value > cutoff:
            samples_are_noise.append(1)
        else:
            samples_are_noise.append(0)

        # Count the number of noise points

    noise_log["noise_samples"] = sum(samples_are_noise)
    noise_log["total_valid_samples"] = len(samples_are_noise)

    _noise = sum(samples_are_noise)

    return _noise, noise_log


def EXPERIMENTAL_signal_from_samples(noise_samples, xmap, cutoff):
    signal_log = {}
    samples_are_sginal = []
    for sample in noise_samples:
        pos = gemmi.Position(*sample)

        value = xmap.interpolate_value(pos)

        # Check if they are over cutoff
        if value > cutoff:
            samples_are_sginal.append(1)
        else:
            samples_are_sginal.append(0)

    signal_log["signal_samples"] = sum(samples_are_sginal)
    signal_log["signal_samples_signal"] = sum([_sample for _sample in samples_are_sginal if _sample > 0])
    signal_log["signal_samples_noise"] = sum([_sample for _sample in samples_are_sginal if _sample == 0])
    signal_log["total_valid_samples"] = len(samples_are_sginal)

    _signal = sum(samples_are_sginal)

    return _signal, signal_log


def EXPERIMENTAL_penalty_from_samples(samples, xmap, cutoff):
    samples_scores = []
    for sample in samples:
        pos = gemmi.Position(*sample)

        value = xmap.interpolate_value(pos)

        # Check if they are over cutoff
        if value < cutoff:
            samples_scores.append(1)

    penalty = sum(samples_scores)

    return penalty



def EXPERIMENTAL_score_structure_signal_to_noise_density(
        structure, xmap,
        cutoff=2.0,
        radius_inner_0=0.0,
        radius_inner_1=0.5,
        radius_outer_0=1.2,
        radius_outer_1=1.5,
):
    rescore_log = {
        "cutoff": float(cutoff),
        "radius_inner_0": float(radius_inner_0),
        "radius_inner_1": float(radius_inner_1),
        "radius_outer_0": float(radius_outer_0),
        "radius_outer_1": float(radius_outer_1),
    }

    loci = get_loci(structure)
    rescore_log["loci"] = [(float(pos.x), float(pos.y), float(pos.z))
                           for pos
                           in loci]
    rescore_log["num_loci"] = len(loci)

    # Get sample points
    positions_array, samples_array = get_samples(loci)
    assert samples_array.shape[0] != 0
    assert samples_array.shape[1] == 3
    assert positions_array.shape[0] != 0

    # Get distances
    distances = get_sample_distances(positions_array, samples_array)

    # Get structure sample points
    structure_samples = np.array(
        [
            (float(pos.x), float(pos.y), float(pos.z))
            for pos
            in loci
        ]
    )

    # Get signal samples: change radius until similar number of points
    signal_samples = truncate_samples(samples_array, distances, radius_inner_0, radius_inner_1)
    rescore_log["signal_samples_shape"] = int(signal_samples.shape[0])

    # Get noise samples
    noise_samples_dict = {}
    for radii in np.linspace(radius_outer_0, radius_outer_1, num=15):
        noise_samples_dict[radii] = truncate_samples(samples_array, distances, radius_outer_0, radii)

    selected_radii = min(
        noise_samples_dict,
        key=lambda _radii: np.abs(noise_samples_dict[_radii].size - signal_samples.size)
    )
    rescore_log["selected_radii"] = selected_radii

    noise_samples = truncate_samples(samples_array, distances, radius_outer_0, selected_radii)
    rescore_log["noise_samples_shape"] = int(noise_samples.shape[0])

    # Getfraction of nearby points that are noise
    _noise, noise_log = EXPERIMENTAL_noise_from_samples(noise_samples, xmap, 0.5)
    rescore_log["noise"] = _noise
    rescore_log["noise_log"] = noise_log

    # Get fraction of bonds/atoms which are signal
    # _signal, signal_log = EXPERIMENTAL_signal_from_samples(structure_samples, xmap, 0.5)
    _signal, signal_log = EXPERIMENTAL_signal_from_samples(signal_samples, xmap, 0.5)
    rescore_log["signal"] = _signal
    rescore_log["signal_log"] = signal_log

    # return (1 - _noise) * _signal, rescore_log

    # TODO: remove if doesn't work
    signal_overlapping_protein_penalty = EXPERIMENTAL_penalty_from_samples(signal_samples, xmap, -0.5)
    rescore_log["penalty"] = signal_overlapping_protein_penalty


    print(f"\t\t\tSignal {_signal} / {len(signal_samples)} Noise {_noise} / {len(noise_samples)} Penalty"
          f" {signal_overlapping_protein_penalty} / {len(signal_samples)}")


    # _score = ((_signal / len(signal_samples)) - np.sqrt(_noise / len(noise_samples))) - np.sqrt(
    #     signal_overlapping_protein_penalty / len(signal_samples))

    _score = (_signal / len(signal_samples)) - (_noise / len(noise_samples)) - (signal_overlapping_protein_penalty /
                                                                                len(signal_samples))

    return _score, rescore_log



def event_map_to_contour_score_map(
        dataset,
        event: EventInterface,
        event_map_grid,
        protein_score=-1.0,
        protein_event_overlap_score=0.0,
):
    # Mask protein
    # if debug >= Debug.PRINT_SUMMARIES:
    #     print("\t\tMasking protein...")
    inner_mask_grid = gemmi.Int8Grid(*[event_map_grid.nu, event_map_grid.nv, event_map_grid.nw])
    inner_mask_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    inner_mask_grid.set_unit_cell(event_map_grid.unit_cell)
    for atom in dataset.structure.protein_atoms():
        pos = atom.pos
        inner_mask_grid.set_points_around(pos,
                                          radius=1.25,
                                          value=1,
                                          )

    outer_mask_grid = gemmi.Int8Grid(*[event_map_grid.nu, event_map_grid.nv, event_map_grid.nw])
    outer_mask_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    outer_mask_grid.set_unit_cell(event_map_grid.unit_cell)
    for atom in dataset.structure.protein_atoms():
        pos = atom.pos
        outer_mask_grid.set_points_around(pos,
                                          radius=6.0,
                                          value=1,
                                          )



    # Get event mask
    event_mask_grid = gemmi.Int8Grid(*[event_map_grid.nu, event_map_grid.nv, event_map_grid.nw])
    event_mask_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    event_mask_grid.set_unit_cell(event_map_grid.unit_cell)

    for x,y,z in event.native_positions:
        pos = gemmi.Position(x,y,z)
        event_mask_grid.set_points_around(pos,
                                          radius=1.0,
                                          value=1,
                                          )


    #
    event_mask_int_array = np.array(
        event_mask_grid,
        copy=False,
        dtype=np.int8,
    )
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

    # Event mask
    event_mask = np.zeros(inner_mask_int_array.shape, dtype=bool)
    event_mask[np.nonzero(event_mask_int_array)] = True
    inner_mask = np.zeros(inner_mask_int_array.shape, dtype=bool)
    inner_mask[np.nonzero(inner_mask_int_array)] = True
    outer_mask = np.zeros(inner_mask_int_array.shape, dtype=bool)
    outer_mask[np.nonzero(outer_mask_int_array)] = True

    #
    event_map_grid_array = np.array(event_map_grid,
                                              copy=False,
                                              )

    # Mask the protein except at event sites with a penalty
    event_map_grid_array[inner_mask & (~event_mask)] = protein_score

    # Mask the protein-event overlaps with zeros
    event_map_grid_array[inner_mask & event_mask] = protein_event_overlap_score

    return event_map_grid

def score_structure_contour(
        optimised_structure,
        zmap_grid,
        res,
        rate,
        structure_map_high_cut=0.6
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
    approximate_structure_map_high_indicies = approximate_structure_map_array > structure_map_high_cut
    num_structure_map_high_indicies = np.sum(approximate_structure_map_high_indicies)
    num_outer_mask_indicies = np.sum(outer_mask_array)

    # Generate the outer mask with about as many indicies as the approximate structure map
    outer_mask_cut = structure_map_high_cut
    outer_mask_indicies = (approximate_structure_map_array>outer_mask_cut) & (~approximate_structure_map_high_indicies)
    # print(f"{outer_mask_cut} : {np.sum(outer_mask_indicies)}")
    while np.sum(outer_mask_indicies) < num_structure_map_high_indicies:
        outer_mask_cut -= 0.025
        outer_mask_indicies = (approximate_structure_map_array > outer_mask_cut) & (~approximate_structure_map_high_indicies)
        # print(f"{outer_mask_cut} : {np.sum(outer_mask_indicies)}")

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
    for cutoff in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.8, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8,
                   2.9, 3.0]:
        noise_percent = np.sum(event_map_array[outer_mask_indexes] > cutoff) / np.sum(outer_mask_array)
        signal = np.sum(event_map_array[mask_indicies] > cutoff)
        score = signal - (np.sum(inner_mask_int_array) * noise_percent)
        scores[float(cutoff)] = int(score)

    scores_from_calc = {}
    noises_from_calc = {}
    signals_from_calc = {}

    # Apply size correction
    num_structure_heavy_atoms = 0.0
    for model in optimised_structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element.name != "H":
                        num_structure_heavy_atoms += 1.0

    for cutoff in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.8, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8,
                   2.9, 3.0]:
        event_map_high_indicies = event_map_array[approximate_structure_map_high_indicies] > cutoff
        # signal = np.sum(event_map_high_indicies)
        signal = np.sum(event_map_high_indicies)
        signal_percent = signal / num_structure_map_high_indicies
        signals_from_calc[float(cutoff)] = float(signal_percent)
        noise_percent = np.sum(event_map_array[outer_mask_indexes] > cutoff) / num_outer_mask_indicies
        noises_from_calc[float(cutoff)] = float(noise_percent)
        # score = signal - (np.sum(approximate_structure_map_array > structure_map_high_cut) * noise_percent)
        # score = signal_percent - noise_percent
        if signal_percent > 0.5:
            score = (signal_percent-noise_percent)*math.sqrt(num_structure_heavy_atoms)
        else:
            score = -0.01

        # scores_from_calc[float(cutoff)] = int(score)
        scores_from_calc[float(cutoff)] = float(score)







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
        "scores_from_calc": scores_from_calc,
        "signals": signals_from_calc,
        "noises": noises_from_calc,
        "Approximate structure map array size": approximate_structure_map_array.size,
        "Event map array size": event_map_array.size,
        "Num structure map high": (approximate_structure_map_array > 1.5).shape
    }


