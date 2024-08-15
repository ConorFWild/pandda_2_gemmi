import typing

import numpy as np
import gemmi
import torch

from .interfaces import *


class SampleFrame(SampleFrameI):
    def __init__(self, transform: TransformI, spacing: typing.Tuple[int, int, int]):
        self.transform: TransformI = transform
        self.spacing: typing.Tuple[int, int, int] = spacing

    def __call__(self, grid: GridI, scale=False) -> np.array:
        arr = np.zeros(self.spacing, dtype=np.float32)
        grid.interpolate_values(arr, self.transform)
        narr = np.copy(arr)
        if scale:
            std = np.std(arr)
            if np.abs(std) > 0.0000001:
                narr = (arr - np.mean(arr)) / std

        return narr


def iterate_atoms(structure: StructureI, hs: bool = False) -> typing.Iterable[AtomI]:
    for model in structure:
        for chain in model:
            for res in chain:
                for atom in res:
                    el = atom.element.name
                    if (not hs) & (el == "H"):
                        continue
                    yield atom


def grid_from_template(template: GridI) -> GridI:
    mask = gemmi.FloatGrid(template.nu, template.nv, template.nw)
    mask.spacegroup = gemmi.find_spacegroup_by_name("P1")
    uc = template.unit_cell
    mask.set_unit_cell(gemmi.UnitCell(uc.a, uc.b, uc.c, uc.alpha, uc.beta, uc.gamma))

    return mask


def get_ligand_mask(ligand: StructureI, template: GridI, radius=1.0) -> GridI:
    mask = grid_from_template(template)

    # Get the mask
    for atom in iterate_atoms(ligand, ):
        pos = atom.pos
        mask.set_points_around(
            pos,
            radius=radius,
            value=1.0,
        )

    return mask


def get_structure_array(st: StructureI):
    poss = []
    for atom in iterate_atoms(st):
        pos = atom.pos
        poss.append([pos.x, pos.y, pos.z])

    return np.array(poss)


def get_structure_centroid(st: StructureI):
    return np.mean(get_structure_array(st), axis=0)


def transform_from_arrays(vec: np.array, mat: np.array) -> TransformI:
    tr = gemmi.Transform()
    tr.vec.fromlist(vec.tolist())
    tr.mat.fromlist(mat.tolist())

    return tr


def load_model_from_checkpoint(path, model):
    # From https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3
    ckpt = torch.load(path, map_location='cpu')
    pretrained_dict = ckpt['state_dict']
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)
    return model


def set_structure_mean(st, centroid):
    st_centroid = get_structure_centroid(st)

    st_clone = st.clone()

    for atom in iterate_atoms(st_clone):
        inital_pos = atom.pos
        atom.pos = gemmi.Position(
            (inital_pos.x - st_centroid[0]) + centroid[0],
            (inital_pos.y - st_centroid[1]) + centroid[1],
            (inital_pos.z - st_centroid[2]) + centroid[2],
        )

    return st_clone
