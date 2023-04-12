import gemmi

from ..interfaces import *

from .dmap import DMap
from .sparse_dmap import SparseDMap
from .sparse_dmap_stream import SparseDMapStream
from .smooth_reflections import SmoothReflections
from .truncate_reflections import TruncateReflections


def load_dmap(path):
    m = gemmi.read_ccp4_map(str(path))
    m.setup(float('nan'))
    return m.grid


def save_dmap(dmap, path, centroid=None, reference_frame: DFrameInterface = None, radius=15.0):
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = dmap
    ccp4.update_ccp4_header(2, True)
    if reference_frame:
        box = gemmi.FractionalBox()
        cart_max = centroid + radius
        cart_min = centroid - radius
        unit_cell = reference_frame.get_grid().unit_cell
        frac_max = unit_cell.fractionalize(gemmi.Position(*cart_max))
        frac_min = unit_cell.fractionalize(gemmi.Position(*cart_min))
        box.extend(frac_max)
        box.extend(frac_min)
        ccp4.set_extent(box)
    ccp4.write_ccp4_map(str(path))
