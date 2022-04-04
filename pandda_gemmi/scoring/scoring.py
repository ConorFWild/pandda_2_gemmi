from __future__ import annotations

import os
import dataclasses
import subprocess
from pathlib import Path
import json

from typing import *

import fire
import numpy as np
import gemmi
import ray

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

    print(f"\t\t\tSignal {_signal} / {len(structure_samples)} Noise {_noise} / {len(noise_samples)} Penalty"
          f" {signal_overlapping_protein_penalty} / {len(signal_samples)}")

    # _score = ((_signal / len(signal_samples)) - np.sqrt(_noise / len(noise_samples))) - np.sqrt(
    #     signal_overlapping_protein_penalty / len(signal_samples))

    _score = (_signal / len(signal_samples)) - (_noise / len(noise_samples)) - (signal_overlapping_protein_penalty /
                                                                                len(signal_samples))

    return _score, rescore_log
