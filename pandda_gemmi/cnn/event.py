import time

import numpy as np
import gemmi
import torch
from torch import nn
from torch.nn import functional as F
import lightning as lt

from .interfaces import *
from .base import transform_from_arrays, SampleFrame, grid_from_template, get_ligand_mask, get_structure_array, copy_map
from .constants import SAMPLE_SIZE, SAMPLE_SPACING
from .resnet import resnet10


def get_sample_frame_from_event(event: EventI, sample_size, sample_spacing) -> SampleFrameI:
    return SampleFrame(
        transform_from_arrays(
            event.centroid - (0.5*sample_spacing*np.array(sample_size)),
            np.eye(3) * sample_spacing,
        ),
        SAMPLE_SIZE,
    )


class Event(EventI):
    def __init__(self, centroid: np.array):
        self.centroid = centroid


def mask_xmap_radial(xmap: GridI, x: float, y: float, z: float, radius: float = 5.5) -> GridI:
    mask = grid_from_template(xmap)
    pos = gemmi.Position(x, y, z)
    mask.set_points_around(
        pos,
        radius=radius,
        value=1.0,
    )

    mask_arr = np.array(mask, copy=False)
    xmap_arr = np.array(xmap, copy=False)

    xmap_arr[:, :, :] = xmap_arr[:, :, :] * mask_arr[:, :, :]
    return xmap


class LitEventScoring(lt.LightningModule):
    def __init__(self, ):
        super().__init__()
        self.z_encoder = resnet10(num_classes=2, num_input=2, headless=True).float()
        self.mol_encoder = resnet10(num_classes=2, num_input=1, headless=True).float()
        self.fc = nn.Sequential(
            nn.Linear(512, 2),
        )

    def forward(self, z, m, ):
        mol_encoding = self.mol_encoder(m)
        z_encoding = self.z_encoder(z)

        full_encoding = z_encoding * F.hardtanh(mol_encoding, min_val=-1.0, max_val=1.0)

        score = F.softmax(self.fc(full_encoding))

        print(score)

        return float(score[0][1])


class EventScorer:

    def __init__(self, model):
        self.model = model

    def __call__(self, event: EventI, ligand_conformation: StructureI, zmap: GridI, xmap: GridI) -> float:
        # Get the sample frame
        sample_frame = get_sample_frame_from_event(event, SAMPLE_SIZE, SAMPLE_SPACING)
        print(f'\t\t{sample_frame.transform.vec.tolist()}')
        print(f'\t\t{sample_frame.transform.mat.tolist()}')
        print(get_structure_array(ligand_conformation))

        # Cut the xmap
        x, y, z = event.centroid
        cut_xmap = mask_xmap_radial(copy_map(xmap), x, y, z)
        xmap_array = np.array(cut_xmap)
        print(f'Cut xmap sum: {np.sum(xmap_array)} {np.min(xmap_array)} {np.max(xmap_array)}')

        # Get the xmap sample
        xmap_sample = sample_frame(cut_xmap, scale=True)

        # Get the zmap sample
        zmap_sample = sample_frame(zmap, scale=True)

        # Get the ligand mask sample
        ligand_mask = get_ligand_mask(ligand_conformation, zmap)
        ligand_mask_sample = sample_frame(ligand_mask, scale=False)

        # Get the ligand mask
        # ligand_mask_sample_bin = np.copy(ligand_mask_sample)
        # ligand_mask_sample_bin[ligand_mask_sample_bin <= 0.0] = 0.0
        # ligand_mask_sample_bin[ligand_mask_sample_bin > 0.0] = 1.0

        # Run the model
        map_array = np.stack(
                    [
                        zmap_sample,
                        xmap_sample,
                    ]
                )[np.newaxis,:]
        mol_array = np.stack(
                [
                    ligand_mask_sample
                ]
            )[np.newaxis,:]
        return self.model(
            torch.from_numpy(map_array
                ),
            torch.from_numpy(mol_array)
        )
