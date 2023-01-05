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
# import ray

from pandda_gemmi.analyse_interface import *
from pandda_gemmi import constants
from pandda_gemmi.analyse_interface import GetAutobuildResultInterface

# from pandda_gemmi.constants import *
# from pandda_gemmi.python_types import *
# from pandda_gemmi.common import Dtag, EventID, EventIDX
# from pandda_gemmi.fs import PanDDAFSModel
from pandda_gemmi.dataset import (StructureFactors, Structure, Reflections, Dataset, ResidueID, Datasets,
                                  Resolution, Reference)
# from pandda_gemmi.shells import Shell
# from pandda_gemmi.edalignment import Alignment, Alignments, Transform, Grid, Partitioning, Xmap
# from pandda_gemmi.model import Zmap, Model
from pandda_gemmi.event import Event
from pandda_gemmi.scoring.scoring import event_map_to_contour_score_map, score_structure_contour
from pandda_gemmi.autobuild.cif import generate_cif, generate_cif_grade, generate_cif_grade2


@dataclasses.dataclass()
class AutobuildResult:
    status: bool
    paths: List[str]
    scores: Dict[str, float]
    cif_path: str
    selected_fragment_path: Optional[str]
    command: str

    def log(self):
        return self.command


def execute(command: str):
    p = subprocess.Popen(command,
                         shell=True,
                         env=os.environ.copy(),
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         )

    stdout, stderr = p.communicate()

    print(stdout, stderr)


@dataclasses.dataclass()
class Coord:
    x: float
    y: float
    z: float


# #####################
# # Truncate Model
# #####################

def get_pdb(pdb_file: Path) -> gemmi.Structure:
    structure: gemmi.Structure = gemmi.read_structure(str(pdb_file))
    return structure


def save_pdb_file(masked_pdb: gemmi.Structure, path: Path) -> Path:
    masked_pdb.write_minimal_pdb(str(path))

    return path


def get_masked_pdb(pdb: gemmi.Structure, coord: Coord, radius: float = 8.0) -> gemmi.Structure:
    event_centoid = gemmi.Position(
        coord.x,
        coord.y,
        coord.z,
    )

    ns = gemmi.NeighborSearch(pdb[0], pdb.cell, radius).populate()

    marks = ns.find_atoms(event_centoid, radius=radius)
    # cras = [mark.to_cra(pdb[0]) for mark in marks]
    for mark in marks:
        mark.to_cra(pdb[0]).atom.flag = 's'

    new_structure = gemmi.Structure()

    for model_i, model in enumerate(pdb):
        new_model = gemmi.Model(model.name)
        new_structure.add_model(new_model, pos=-1)

        for chain_i, chain in enumerate(model):
            new_chain = gemmi.Chain(chain.name)
            new_structure[model_i].add_chain(new_chain, pos=-1)

            for residue_i, residue in enumerate(chain):
                new_residue = gemmi.Residue()
                new_residue.name = residue.name
                new_residue.seqid = residue.seqid
                new_residue.subchain = residue.subchain
                new_residue.label_seq = residue.label_seq
                new_residue.het_flag = residue.het_flag
                new_structure[model_i][chain_i].add_residue(new_residue, pos=-1)

                for atom_i, atom in enumerate(residue):
                    pos = atom.pos
                    # if pos.dist(event_centoid) > radius:
                    if atom.flag == "s":
                        continue
                    new_structure[model_i][chain_i][residue_i].add_atom(atom, pos=-1)

    for model_i, model in enumerate(pdb):
        pdb.add_model(new_structure[model_i], pos=-1)
        del pdb[0]

    return pdb


def truncate_model(model_path: Path, coords: Coord, out_dir: Path):
    # Get pdb
    pdb: gemmi.Structure = get_pdb(model_path)

    # Truncate
    masked_pdb = get_masked_pdb(pdb, coords)

    # Write
    masked_pdb_file = save_pdb_file(masked_pdb,
                                    out_dir / constants.MASKED_PDB_FILE,
                                    )

    # Return
    return masked_pdb_file


# #####################
# # Truncate event map
# #####################


def get_cut_out_event_map(
        event_map: gemmi.FloatGrid,
        coords: List[Tuple[float, float, float]],
        radius: float = 3.0,
) -> gemmi.FloatGrid:
    xmap_array = np.array(event_map, copy=True)

    mask_grid = gemmi.Int8Grid(*xmap_array.shape)
    # print(f"Spacegroup: {mask_grid.spacegroup.xhm()}")
    # mask_grid.spacegroup = gemmi.find_spacegroup_by_name("P 21 21 21")  #  gemmi.find_spacegroup_by_name("P 1")#event_map.spacegroup
    mask_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")  # event_map.spacegroup
    mask_grid_array = np.array(mask_grid)

    # print(f"Grid size: {mask_grid.size}")
    mask_grid.set_unit_cell(event_map.unit_cell)

    for position_python in coords:
        position = gemmi.Position(*position_python)
        mask_grid.set_points_around(position,
                                    radius=radius,
                                    value=1,
                                    )
    mask_grid.symmetrize_max()

    mask_array = np.array(mask_grid, copy=False, dtype=np.int8)

    new_grid = gemmi.FloatGrid(*xmap_array.shape)
    new_grid.spacegroup = event_map.spacegroup  # gemmi.find_spacegroup_by_name("P 1")
    new_grid.set_unit_cell(event_map.unit_cell)

    new_grid_array = np.array(new_grid, copy=False)

    new_grid_array[np.nonzero(mask_array)] = xmap_array[np.nonzero(mask_array)]
    # new_grid.symmetrize_max()

    return new_grid


def get_event_map(event_map_file: Path) -> gemmi.FloatGrid:
    m = gemmi.read_ccp4_map(str(event_map_file))

    # m.grid.spacegroup = gemmi.find_spacegroup_by_name("P 21 21 21")
    m.grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")

    m.setup()

    grid_array = np.array(m.grid, copy=True)

    new_grid = gemmi.FloatGrid(*grid_array.shape)
    new_grid.spacegroup = m.grid.spacegroup  # gemmi.find_spacegroup_by_name("P 1")
    new_grid.set_unit_cell(m.grid.unit_cell)

    new_grid_array = np.array(new_grid, copy=False)
    new_grid_array[:, :, :] = grid_array[:, :, :]

    return new_grid


def save_xmap_dep(cut_out_event_map,
                  path,
                  ):
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = cut_out_event_map

    ccp4.grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")

    ccp4.update_ccp4_header(2, True)
    ccp4.write_ccp4_map(str(path))

    return path


def save_xmap(event_map,
              path,
              ):
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = event_map

    ccp4.grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")

    ccp4.update_ccp4_header(2, True)
    ccp4.write_ccp4_map(str(path))

    return path


def truncate_xmap(xmap_path: Path, coords: List[Tuple[float, float, float]], out_dir: Path):
    event_map: gemmi.FloatGrid = get_event_map(xmap_path)

    # Cut out events:
    cut_out_event_map: gemmi.FloatGrid = get_cut_out_event_map(event_map, coords)

    # Save cut out event
    cut_out_event_map_file: Path = save_xmap(cut_out_event_map,
                                             out_dir / constants.TRUNCATED_EVENT_MAP_FILE,
                                             )

    return cut_out_event_map_file


# #####################
# # Cut out map
# #####################
def get_ccp4_map(xmap_path):
    m = gemmi.read_ccp4_map(str(xmap_path))
    m.grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")

    m.setup()

    return m


def get_bounding_box(event_map: gemmi.FloatGrid, coord: Coord, radius: float = 8.0,
                     margin: float = 5.0) -> gemmi.FloatGrid:
    event_centroid = gemmi.Position(coord.x, coord.y, coord.z)

    box_lower_bound = gemmi.Position(float(coord.x) - radius, float(coord.y) - radius, float(coord.z) - radius)
    box_upper_bound = gemmi.Position(float(coord.x) + radius, float(coord.y) + radius, float(coord.z) + radius)

    box_lower_bound_fractional = event_map.grid.unit_cell.fractionalize(box_lower_bound)
    box_upper_bound_fractional = event_map.grid.unit_cell.fractionalize(box_upper_bound)

    box = gemmi.FractionalBox()

    box.extend(box_lower_bound_fractional)
    box.extend(box_upper_bound_fractional)

    return box


def save_cut_xmap(event_ccp4,
                  bounding_box,
                  path,
                  ):
    event_ccp4.set_extent(bounding_box)
    event_ccp4.write_ccp4_map(str(path))

    return path


def cut_out_xmap(xmap_path: Path, coord: Coord, out_dir: Path):
    ccp4_map = get_ccp4_map(xmap_path)

    # Cut out events:
    bounding_box = get_bounding_box(ccp4_map, coord)

    # Save cut out event
    cut_out_event_map_file: Path = save_cut_xmap(ccp4_map,
                                                 bounding_box,
                                                 out_dir / constants.CUT_EVENT_MAP_FILE,
                                                 )

    return cut_out_event_map_file




# #####################
# # rhofit
# #####################

def rhofit(truncated_model_path: Path, truncated_xmap_path: Path, mtz_path: Path, cif_path: Path,
           out_dir: Path, cut: float = 2.0, debug: Debug = Debug.DEFAULT
           ):
    # Make rhofit commands
    pandda_rhofit = str(Path(__file__).parent / constants.PANDDA_RHOFIT_SCRIPT_FILE)

    rhofit_command: str = constants.RHOFIT_COMMAND.format(
        pandda_rhofit=pandda_rhofit,
        event_map=str(truncated_xmap_path),
        mtz=str(mtz_path),
        pdb=str(truncated_model_path),
        cif=str(cif_path),
        out_dir=str(out_dir),
        cut=cut,
    )

    # Execute job script
    execute(rhofit_command)

    return rhofit_command


def rhofit_to_coord(truncated_model_path: Path, truncated_xmap_path: Path, mtz_path: Path, cif_path: Path,
                    out_dir: Path,
                    coord: Coord,
                    cut: float = 2.0,
                    debug=False
                    ):
    # Make rhofit commands
    pandda_rhofit = str(Path(__file__).parent / constants.PANDDA_RHOFIT_SCRIPT_FILE)

    rhofit_command: str = constants.RHOFIT_COMMAND_COORD.format(
        pandda_rhofit=pandda_rhofit,
        event_map=str(truncated_xmap_path),
        mtz=str(mtz_path),
        pdb=str(truncated_model_path),
        cif=str(cif_path),
        out_dir=str(out_dir),
        x=coord.x,
        y=coord.y,
        z=coord.z,
        cut=cut,
    )

    # Execute job script
    execute(rhofit_command)

    return rhofit_command


# #####################
# Rescore
# #####################

def score_structure(structure, xmap):
    unit_cell = xmap.unit_cell

    mask = gemmi.FloatGrid(
        xmap.nu, xmap.nv, xmap.nw
    )

    mask.set_unit_cell(unit_cell)
    mask.spacegroup = gemmi.find_spacegroup_by_name("P 1")

    for model in structure:
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
                            mask.set_points_around(new_pos, 0.75, 1.0)

    mask_array = np.array(mask)

    xmap_array = np.array(xmap)

    truncated_xmap_mask = xmap_array > 1.25

    score = np.sum(truncated_xmap_mask * mask_array)

    return float(score)


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


def signal(positions, xmap, cutoff):
    signal_log = {}

    signal_list = []
    for position in positions:
        val = xmap.interpolate_value(position)
        if val > cutoff:
            signal_list.append(1)
        else:
            signal_list.append(0)

    signal_log["signal_points"] = sum(signal_list)
    signal_log["num_points"] = len(signal_list)

    # _signal = sum(signal_list) * (sum(signal_list) / len(signal_list))

    # _signal = sum(signal_list) / len(signal_list)
    #
    # _signal = (2*sum(signal_list)) - len(signal_list)

    total_signal = sum(signal_list)

    expectected_signal = len(signal_list)

    fractional_signal = total_signal / expectected_signal

    if fractional_signal > 0.4:
        _signal = sum(signal_list)

    else:
        _signal = -10000.0

    return _signal, signal_log


def noise(positions, xmap, cutoff, radius, num_samples=100):
    noise_log = {}

    # for each position
    positions_list = []
    samples_arrays_list = []
    for position in positions:
        # Get some random vectors
        position_array = np.array([position.x, position.y, position.z]).reshape((1, 3))
        positions_list.append(position_array)

        deltas_array = (np.random.rand(num_samples, 3) - 0.5) * 2

        # Scale them to radius
        scaled_deltas_array = (radius / np.linalg.norm(deltas_array, axis=1)).reshape((num_samples, 1)) * deltas_array

        # Add them to pos
        samples_array = position_array + scaled_deltas_array
        samples_arrays_list.append(samples_array)

    positions_array = np.vstack(positions_list)
    samples_arrays_array = np.vstack(samples_arrays_list)

    noise_log["positions_array"] = positions_array.tolist()
    noise_log["samples_array"] = samples_arrays_array.tolist()

    # Get rid of ones near other points
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(positions_array)
    distances, indices = nbrs.kneighbors(samples_arrays_array)
    valid_samples = samples_arrays_array[(distances > 0.95 * radius).flatten(), :]

    noise_log["valid_samples_array"] = valid_samples.tolist()

    # Interpolate at ones left over
    samples_are_noise = []
    for sample in valid_samples:
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

    # _noise = sum(samples_are_noise) / len(samples_are_noise)

    _noise = sum(samples_are_noise)

    return _noise, noise_log


def get_samples(positions,
                sample_density=10,
                buffer=2.0,
                # radius_inner_0=0.0,
                # radius_inner_1=0.2,
                # radius_outer_0=1.2,
                # radius_outer_1=1.5,
                # radius=1.3
                ):
    positions_array_list = []
    # for position in positions:
    #     # Get some random vectors
    #     position_array = np.array([position.x, position.y, position.z]).reshape((1, 3))
    #     positions_list.append(position_array)
    #
    #     deltas_array = (np.random.rand(num_samples, 3) - 0.5) * 2
    #
    #     # Scale them to radius
    #     scaled_deltas_array = ((np.random.rand(num_samples) * radius) / np.linalg.norm(deltas_array, axis=1)).reshape((
    #         num_samples, 1)) * deltas_array
    #
    #     # Add them to pos
    #     samples_array = position_array + scaled_deltas_array
    #     samples_arrays_list.append(samples_array)
    #
    # positions_array = np.vstack(positions_list)
    # samples_arrays_array = np.vstack(samples_arrays_list)

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


def noise_from_samples(noise_samples, xmap, cutoff):
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


def signal_from_samples(noise_samples, xmap, cutoff):
    signal_log = {}
    samples_are_sginal = []
    for sample in noise_samples:
        pos = gemmi.Position(*sample)

        value = xmap.interpolate_value(pos)

        # Check if they are over cutoff
        if value > cutoff:
            samples_are_sginal.append(1)
        else:
            samples_are_sginal.append(-1)

    signal_log["signal_samples"] = sum(samples_are_sginal)
    signal_log["signal_samples_signal"] = sum([_sample for _sample in samples_are_sginal if _sample > 0])
    signal_log["signal_samples_noise"] = sum([_sample for _sample in samples_are_sginal if _sample < 0])
    signal_log["total_valid_samples"] = len(samples_are_sginal)

    _signal = sum(samples_are_sginal)

    return _signal, signal_log


def penalty_from_samples(samples, xmap, cutoff):
    samples_scores = []
    for sample in samples:
        pos = gemmi.Position(*sample)

        value = xmap.interpolate_value(pos)

        # Check if they are over cutoff
        if value < cutoff:
            samples_scores.append(1)

    penalty = sum(samples_scores)

    return penalty


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


def score_structure_signal_to_noise_density(
        structure, xmap,
        cutoff=2.0,
        radius_inner_0=0.0,
        radius_inner_1=0.5,
        radius_outer_0=1.2,
        radius_outer_1=1.5,
        debug: Debug = Debug.DEFAULT
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
    _noise, noise_log = noise_from_samples(noise_samples, xmap, cutoff)
    rescore_log["noise"] = _noise
    rescore_log["noise_log"] = noise_log

    # Get fraction of bonds/atoms which are signal
    _signal, signal_log = signal_from_samples(signal_samples, xmap, cutoff)
    rescore_log["signal"] = _signal
    rescore_log["signal_log"] = signal_log

    # return (1 - _noise) * _signal, rescore_log

    # TODO: remove if doesn't work
    signal_overlapping_protein_penalty = penalty_from_samples(signal_samples, xmap, -10)
    noise_overlapping_protein_penalty = penalty_from_samples(
        noise_samples,
        xmap, -10)
    ligand_overlapping_protein_penalty = signal_overlapping_protein_penalty + noise_overlapping_protein_penalty

    if debug >= Debug.PRINT_NUMERICS:
        print(f"\t\t\tSignal {_signal} Noise {_noise} Penalty {ligand_overlapping_protein_penalty}")

    _score = ((_signal - _noise) - ligand_overlapping_protein_penalty)

    return _score, rescore_log


def EXPERIMENTAL_score_structure_signal_to_noise_density(
        structure, xmap,
        cutoff=2.0,
        radius_inner_0=0.0,
        radius_inner_1=0.5,
        radius_outer_0=1.2,
        radius_outer_1=1.5,
        debug: Debug = Debug.DEFAULT
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
    _signal, signal_log = EXPERIMENTAL_signal_from_samples(structure_samples, xmap, 0.5)
    rescore_log["signal"] = _signal
    rescore_log["signal_log"] = signal_log

    # return (1 - _noise) * _signal, rescore_log

    # TODO: remove if doesn't work
    signal_overlapping_protein_penalty = EXPERIMENTAL_penalty_from_samples(signal_samples, xmap, -0.5)
    if debug >= Debug.PRINT_NUMERICS:
        print(f"\t\t\tSignal {_signal} / {len(structure_samples)} Noise {_noise} / {len(noise_samples)} Penalty"
              f" {signal_overlapping_protein_penalty} / {len(signal_samples)}")

    _score = ((_signal / len(structure_samples)) - np.sqrt(_noise / len(noise_samples))) - np.sqrt(
        signal_overlapping_protein_penalty / len(signal_samples))

    return _score, rescore_log


def score_structure_signal_to_noise(structure, xmap, cutoff=2.0, radius=1.2):
    rescore_log = {
        "cutoff": float(cutoff),
        "radius": float(radius)
    }

    loci = get_loci(structure)
    rescore_log["loci"] = [(float(pos.x), float(pos.y), float(pos.z))
                           for pos
                           in loci]
    rescore_log["num_loci"] = len(loci)

    # Getfraction of nearby points that are noise
    _noise, noise_log = noise(loci, xmap, cutoff, radius)
    rescore_log["noise"] = _noise
    rescore_log["noise_log"] = noise_log

    # Get fraction of bonds/atoms which are signal
    _signal, signal_log = signal(loci, xmap, cutoff)
    rescore_log["signal"] = _signal
    rescore_log["signal_log"] = signal_log

    # return (1 - _noise) * _signal, rescore_log

    return _signal - _noise, rescore_log


def score_structure_path(path: Path, xmap):
    structure = gemmi.read_structure(str(path))
    # score = score_structure(structure, xmap)
    # score, rescore_log = score_structure_signal_to_noise_density(structure, xmap)
    score, rescore_log = EXPERIMENTAL_score_structure_signal_to_noise_density(
        structure, xmap)

    return score, rescore_log


def score_builds(rhofit_dir: Path, score_model_path, xmap_path, zmap_path):
    scores = {}
    rescoring_log = {}

    regex = "Hit_*.pdb"
    rescoring_log["regex"] = regex

    event_map_reference_grid = get_ccp4_map(xmap_path).grid

    event_map_reference_grid_array = np.array(event_map_reference_grid,
                                              copy=False,
                                              )

    score_model = Structure.from_file(score_model_path)
    inner_mask = gemmi.Int8Grid(
        *[event_map_reference_grid.nu, event_map_reference_grid.nv, event_map_reference_grid.nw])
    inner_mask.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    inner_mask.set_unit_cell(event_map_reference_grid.unit_cell)
    for atom in score_model.protein_atoms():
        pos = atom.pos
        inner_mask.set_points_around(pos,
                                     radius=2.0,
                                     value=1,
                                     )

    event_map_reference_grid_array[event_map_reference_grid_array < 2.0] = 0.0
    event_map_reference_grid_array[event_map_reference_grid_array >= 2.0] = 1.0

    # Mask the protein except around the event
    # inner_mask = grid.partitioning.inner_mask
    inner_mask_array = np.array(
        inner_mask,
        copy=False,
        dtype=np.int8,
    )

    z_map_reference_grid = get_ccp4_map(zmap_path).grid

    z_map_reference_grid_array = np.array(z_map_reference_grid,
                                          copy=False,
                                          )

    inner_mask_array[z_map_reference_grid_array > 2.0] = 0.0
    # event_map_reference_grid_array[np.nonzero(inner_mask_array)] = 0.0
    event_map_reference_grid_array[np.nonzero(inner_mask_array)] = -1.0

    for model_path in rhofit_dir.glob(regex):
        score, rescore_log = score_structure_path(model_path, event_map_reference_grid)
        scores[str(model_path)] = score
        rescoring_log[str(model_path)] = rescore_log

    return scores, rescoring_log


def score_builds_contour(
        rhofit_dir: Path,
        score_model_path,
        event_map_path,
        zmap_path,
        dataset: DatasetInterface,
        event: EventInterface,
):
    scores = {}
    rescoring_log = {}

    regex = "Hit_*.pdb"
    rescoring_log["regex"] = regex

    # event_map_reference_grid = get_ccp4_map(xmap_path).grid

    # event_map_reference_grid_array = np.array(event_map_reference_grid,
    #                                           copy=False,
    #                                           )

    # score_model = Structure.from_file(score_model_path)
    # inner_mask = gemmi.Int8Grid(
    #     *[event_map_reference_grid.nu, event_map_reference_grid.nv, event_map_reference_grid.nw])
    # inner_mask.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    # inner_mask.set_unit_cell(event_map_reference_grid.unit_cell)
    # for atom in score_model.protein_atoms():
    #     pos = atom.pos
    #     inner_mask.set_points_around(pos,
    #                                  radius=2.0,
    #                                  value=1,
    #                                  )
    #
    # event_map_reference_grid_array[event_map_reference_grid_array < 2.0] = 0.0
    # event_map_reference_grid_array[event_map_reference_grid_array >= 2.0] = 1.0
    #
    # # Mask the protein except around the event
    # # inner_mask = grid.partitioning.inner_mask
    # inner_mask_array = np.array(
    #     inner_mask,
    #     copy=False,
    #     dtype=np.int8,
    # )

    event_map_reference_grid = get_ccp4_map(event_map_path).grid

    score_grid = event_map_to_contour_score_map(
        dataset,
        event,
        event_map_reference_grid,

    )

    # inner_mask_array[z_map_reference_grid_array > 2.0] = 0.0
    # event_map_reference_grid_array[np.nonzero(inner_mask_array)] = 0.0
    # event_map_reference_grid_array[np.nonzero(inner_mask_array)] = -1.0

    for model_path in rhofit_dir.glob(regex):
        structure = gemmi.read_structure(str(model_path))
        score, rescore_log = score_structure_contour(
            structure,
            score_grid,
            res=dataset.reflections.get_resolution(),
            rate=0.0,
        )

        save_xmap(rescore_log["grid"], rhofit_dir / "structure_map.ccp4")

        del rescore_log["grid"]
        scores[str(model_path)] = score
        rescoring_log[str(model_path)] = rescore_log

    return scores, rescoring_log


def save_score_dictionary(score_dictionary, path):
    with open(str(path), "w") as f:
        json.dump(score_dictionary, f)


def merge_ligand_into_structure_from_paths(receptor_path, ligand_path):
    receptor = get_pdb(receptor_path)
    ligand = get_pdb(ligand_path)

    for receptor_model in receptor:
        for receptor_chain in receptor_model:

            seqid_nums = []
            for receptor_res in receptor_chain:
                num = receptor_res.seqid.num
                seqid_nums.append(num)

            if len(seqid_nums) == 0:
                min_ligand_seqid = 1
            else:
                min_ligand_seqid = max(seqid_nums) + 1

            for model in ligand:
                for chain in model:
                    for residue in chain:
                        residue.seqid.num = min_ligand_seqid

                        receptor_chain.add_residue(residue, pos=-1)

            break
        break

    return receptor


# #####################
# # Core autobuilding
# #####################
# def _autobuild(
#         model: str,
#         xmap: str,
#         mtz: str,
#         cif_path: str,
#         x: float,
#         y: float,
#         z: float,
#         out_dir: str,
# ):
#     # Type all the input variables
#     model_path = Path(model)
#     xmap_path = Path(xmap)
#     mtz_path = Path(mtz)
#     smiles_path = Path(smiles)
#     out_dir = Path(out_dir)
#     coords = Coord(x, y, z)
#
#     # Truncate the model
#     truncated_model_path = truncate_model(model_path, coords, out_dir)
#     print(f"\tTruncated model")
#
#     # Truncate the ed map
#     truncated_xmap_path = truncate_xmap(xmap_path, coords, out_dir)
#     print(f"\tTruncated xmap")
#
#     # Make cut out map
#     cut_out_xmap(xmap_path, coords, out_dir)
#     print(f"\tCut out xmap")
#
#     # Generate the cif
#     cif_path = generate_cif(smiles_path, out_dir, phenix_setup)
#     print(f"\tGenerated cif")
#
#     # Call rhofit
#     rhofit(truncated_model_path, truncated_xmap_path, mtz_path, cif_path, out_dir,
#               phenix_setup, rhofit_setup,)
#     print(f"\tRhofit")
#
#     # Score rhofit builds
#     score_dictionary = score_builds(out_dir / "rhofit", xmap_path)
#     print(f"\tRescored")
#     for path in sorted(score_dictionary, key=lambda _path: score_dictionary[_path]):
#         print(f"\t\t{score_dictionary[path]}: {path}")
#
#     # Write scores
#     save_score_dictionary(score_dictionary, out_dir / "scores.json")
#     print(f"\tSaved scores")
#
#     # Remove the big map
#     print(f"Removing truncated map")
#     os.remove(str(truncated_xmap_path))


# #####################
# # Autobuild from pandda
# #####################

def autobuild_rhofit(dataset: Dataset,
                     event: Event,
                     pandda_fs,
                     cif_strategy,
                     cut: float = 2.0,
                     rhofit_coord: bool = False,
                     debug: Debug = Debug.DEFAULT
                     ):
    # Type all the input variables
    processed_dataset_dir = pandda_fs.processed_datasets[event.event_id.dtag]
    score_model_path = processed_dataset_dir.source_pdb
    score_map_path = pandda_fs.processed_datasets[event.event_id.dtag].event_map_files[event.event_id.event_idx].path
    # build_map_path = pandda_fs.processed_datasets[event.event_id.dtag].z_map_file.path
    zmap_path = pandda_fs.processed_datasets[event.event_id.dtag].z_map_file.path
    build_map_path = pandda_fs.processed_datasets[event.event_id.dtag].event_map_files[event.event_id.event_idx].path
    # score_map_path = pandda_fs.processed_datasets[event.event_id.dtag].z_map_file.path
    out_dir = pandda_fs.processed_datasets[event.event_id.dtag].path / f"{event.event_id.event_idx.event_idx}"
    model_path = processed_dataset_dir.input_pdb
    mtz_path = processed_dataset_dir.input_mtz
    cif_path = pandda_fs.processed_datasets[event.event_id.dtag].source_ligand_cif
    smiles_path = pandda_fs.processed_datasets[event.event_id.dtag].source_ligand_smiles

    try:
        os.mkdir(str(out_dir))
    except Exception as e:
        print(e)

    # Log
    autobuilding_log_file = out_dir / "log.json"
    autobuilding_log = {}

    #
    model_path = Path(model_path)
    build_map_path = Path(build_map_path)
    mtz_path = Path(mtz_path)

    if smiles_path:
        smiles_path = Path(smiles_path)
    if cif_path:
        cif_path = Path(cif_path)

    out_dir = Path(out_dir)
    coord = Coord(
        event.native_centroid[0],
        event.native_centroid[1],
        event.native_centroid[2],
    )
    coords = event.native_positions

    # Truncate the model
    truncated_model_path = truncate_model(model_path, coord, out_dir)

    # Truncate the ed map
    if not rhofit_coord:
        truncated_xmap_path = truncate_xmap(build_map_path, coords, out_dir)

        # Make cut out map
        cut_out_xmap(build_map_path, coord, out_dir)

    # Generate the cif
    if cif_strategy == "default":
        if debug >= Debug.PRINT_SUMMARIES:
            print(f"\t{event.event_id}   Making cif with default strategy using cif {cif_path}")
        if not cif_path:
            return AutobuildResult(
                False,
                [],
                {},
                "",
                "",
                "",
            )

        cif_path = cif_path

    # # Makinf with elbow
    # elif cif_strategy == "elbow":
    #     if debug >= Debug.PRINT_SUMMARIES:
    #         print(f"\t{event.event_id}   Making cif with elbow")
    #
    #     if cif_path:
    #         if debug >= Debug.PRINT_SUMMARIES:
    #             print(f"\t\t{event.event_id}   Making cif with elbow using cif: {cif_path}")
    #
    #         cif_path, pdb_path = generate_cif(
    #             cif_path,
    #             out_dir,
    #         )
    #     elif smiles_path:
    #         if debug >= Debug.PRINT_SUMMARIES:
    #             print(f"\t\t{event.event_id}   Making cif with elbow using smiles: {smiles_path}")
    #
    #         cif_path, pdb_path = generate_cif(
    #             smiles_path,
    #             out_dir,
    #         )
    #
    #     else:
    #         return AutobuildResult(
    #             False,
    #             [],
    #             {},
    #             "",
    #             "",
    #             "",
    #         )
    # # Makinf with grade
    # elif cif_strategy == "grade":
    #     if cif_path:
    #         cif_path = generate_cif_grade(
    #             cif_path,
    #             out_dir,
    #         )
    #     elif smiles_path:
    #         cif_path = generate_cif_grade(
    #             smiles_path,
    #             out_dir,
    #         )
    #
    #     else:
    #         return AutobuildResult(
    #             False,
    #             [],
    #             {},
    #             "",
    #             "",
    #             "",
    #         )
    #
    # # Makinf with grade2
    # elif cif_strategy == "grade2":
    #     # if cif_path:
    #     #     cif_path = generate_cif_grade2(
    #     #         cif_path,
    #     #         out_dir,
    #     #     )
    #     if smiles_path:
    #         cif_path = generate_cif_grade2(
    #             smiles_path,
    #             out_dir,
    #         )
    #
    #     else:
    #         return AutobuildResult(
    #             False,
    #             [],
    #             {},
    #             "",
    #             "",
    #             "",
    #         )
    #
    # else:
    #     raise Exception(f"cif_strategy was somehow set to the invalid value: {cif_strategy}")

    # Call rhofit
    if rhofit_coord:
        if debug >= Debug.PRINT_SUMMARIES:
            print(f"\t{event.event_id}   Using rhofit coord")

        rhofit_command = rhofit_to_coord(truncated_model_path, build_map_path, mtz_path, cif_path, out_dir, coord,
                                         cut, debug)
    else:
        if debug >= Debug.PRINT_SUMMARIES:
            print(f"\t{event.event_id}   Using rhofit without coord")

        rhofit_command = rhofit(truncated_model_path, truncated_xmap_path, mtz_path, cif_path, out_dir, cut, debug)

    autobuilding_log["rhofit_command"] = str(rhofit_command)

    # TODO: See if this change works
    # Score rhofit builds
    # score_dictionary, rescoring_log = score_builds(
    #     out_dir / "rhofit",
    #     score_model_path,
    #     score_map_path,
    #     zmap_path,
    #     dataset,
    # )
    score_dictionary, rescoring_log = score_builds_contour(
        out_dir / "rhofit",
        score_model_path,
        score_map_path,
        zmap_path,
        dataset,
        event,
    )

    autobuilding_log["rescoring_log"] = rescoring_log

    # for path in sorted(score_dictionary, key=lambda _path: score_dictionary[_path]):
    #     print(f"\t\t{score_dictionary[path]}: {path}")

    autobuilding_log["scores"] = {
        str(path): float(score_dictionary[path])
        for path
        in sorted(score_dictionary, key=lambda _path: score_dictionary[_path])
    }

    # Write scores
    save_score_dictionary(score_dictionary, out_dir / "scores.json")

    # Remove the big map
    # os.remove(str(truncated_xmap_path))

    # Select fragment build
    if len(score_dictionary) == 0:
        selected_fragement_path = None
    else:
        selected_fragement_path = max(
            score_dictionary,
            key=lambda _path: score_dictionary[_path],
        )

    with open(autobuilding_log_file, "w") as f:
        json.dump(autobuilding_log, f)

    # return result
    return AutobuildResult(
        True,
        [path for path in score_dictionary],
        score_dictionary,
        cif_path,
        selected_fragement_path,
        rhofit_command
    )


# @ray.remote
# def autobuild_rhofit_ray(dataset: Dataset,
#                          event: Event,
#                          pandda_fs,
#                          cif_strategy,
#                          cut: float = 2.0,
#                          rhofit_coord: bool = False,
#                          debug=False):
#     return autobuild_rhofit(dataset,
#                             event,
#                             pandda_fs,
#                             cif_strategy,
#                             cut,
#                             rhofit_coord,
#                             debug
#                             )

class GetAutobuildResultRhofit(GetAutobuildResultInterface):
    tag = "rhofit"
    def __call__(self,
                 dataset: DatasetInterface,
                 event: EventInterface,
                 pandda_fs: PanDDAFSModelInterface,
                 cif_strategy: str,
                 cut: float,
                 rhofit_coord: bool,
                 debug: Debug) -> AutobuildResultInterface:
        return autobuild_rhofit(dataset, event, pandda_fs, cif_strategy, cut, rhofit_coord, debug)
