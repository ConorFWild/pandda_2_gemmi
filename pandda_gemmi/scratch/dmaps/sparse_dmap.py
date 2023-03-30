import numpy as np

from ..interfaces import *

class SparseDMap:

    def __init__(self, data: np.array):
        self.data = data

    @classmethod
    def from_xmap(cls, xmap: CrystallographicGridInterface, dframe: DFrameInterface):
        xmap_array = np.array(xmap, copy=False)

        data = xmap_array[dframe.mask.indicies]

        return cls(data)

    ...