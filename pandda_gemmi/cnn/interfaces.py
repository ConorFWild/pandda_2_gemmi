import typing

import numpy as np

from typing import Protocol


class EventI(Protocol):
    centroid: np.array


class ElementI(Protocol):
    name: str


class PositionI(Protocol):
    x: float
    y: float
    z: float


class AtomI(Protocol):
    pos: PositionI
    element: ElementI


class ResidueI(Protocol):
    def __iter__(self) -> typing.Iterable[AtomI]:
        ...


class ChainI(Protocol):
    def __iter__(self) -> typing.Iterable[ResidueI]:
        ...


class ModelI(Protocol):
    def __iter__(self) -> typing.Iterable[ChainI]:
        ...


class StructureI(Protocol):
    def __iter__(self) -> typing.Iterable[ModelI]:
        ...


class TransformI(Protocol):
    ...


class UnitCellI(Protocol):
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float


class GridI(Protocol):
    nu: int
    nv: int
    nw: int
    unit_cell: UnitCellI

    def interpolate_values(self, arr: np.array, t: TransformI) -> np.array:
        ...

    def set_points_around(
            self,
            pos: PositionI,
            radius: float = 1.0,
            value: float = 1.0,
    ):
        ...


class SampleFrameI(Protocol):
    def __call__(self, grid: GridI, scale=False) -> np.array:
        ...
