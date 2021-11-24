from __future__ import annotations

import typing
import dataclasses


import numpy as np
import gemmi


@dataclasses.dataclass()
class PositionsArray:
    array: np.ndarray

    @staticmethod
    def from_positions(positions: typing.List[gemmi.Position]):
        accumulator = []
        for position in positions:
            # pos = [position.x, position.y, position.z]
            pos = position
            accumulator.append(pos)

        array = np.array(accumulator)

        return PositionsArray(array)

    def to_array(self):
        return self.array

    def to_positions(self):
        positions = []

        for row in self.array:
            pos = gemmi.Position(row[0],
                                 row[1],
                                 row[2], )
            positions.append(pos)

        return positions
