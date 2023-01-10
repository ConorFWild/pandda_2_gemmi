# Base python
import dataclasses
import time
import pprint
from functools import partial
import os
import json
from typing import Set
import pickle

from pandda_gemmi.pandda_logging import STDOUTManager, log_arguments, PanDDAConsole


console = PanDDAConsole()

printer = pprint.PrettyPrinter()

# Scientific python libraries
# import ray
import numpy as np
import gemmi

## Custom Imports


from pandda_gemmi.analyse_interface import *


def save_array_to_map_file(
        array: NDArrayInterface,
        template: CrystallographicGridInterface,
        path: Path
):
    spacing = [template.nu, template.nv, template.nw]
    unit_cell = template.unit_cell
    grid = gemmi.FloatGrid(spacing[0], spacing[1], spacing[2])
    grid.set_unit_cell(unit_cell)
    grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")

    grid_array = np.array(grid, copy=False)
    grid_array[:, :, :] = array[:, :, :]

    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = grid
    ccp4.update_ccp4_header(2, True)
    ccp4.setup()
    ccp4.write_ccp4_map(str(path))


def save_xmap(
        xmap: XmapInterface,
        path: Path
):
    xmap.xmap.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = xmap.xmap
    ccp4.update_ccp4_header(2, True)
    ccp4.setup()
    ccp4.write_ccp4_map(str(path))


def save_raw_xmap(
        dataset: DatasetInterface,
        path: Path,
        structure_factors,
        sample_rate,
):
    unaligned_xmap: gemmi.FloatGrid = dataset.reflections.transform_f_phi_to_map(structure_factors.f,
                                                                                 structure_factors.phi,
                                                                                 sample_rate=sample_rate,
                                                                                 )
    unaligned_xmap.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = unaligned_xmap
    ccp4.update_ccp4_header(2, True)
    ccp4.setup()
    ccp4.write_ccp4_map(str(path))
