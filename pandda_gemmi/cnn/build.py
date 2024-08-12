import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import lightning as lt

from .interfaces import *
from .base import get_ligand_mask, iterate_atoms, grid_from_template, get_structure_centroid, transform_from_arrays, \
    SampleFrame
from .constants import SAMPLE_SIZE
from .resnet import resnet10


def mask_xmap_ligand(autobuild: StructureI, xmap: GridI, radius=1.0) -> GridI:
    mask = grid_from_template(xmap)
    for atom in iterate_atoms(autobuild):
        pos = atom.pos
        mask.set_points_around(
            pos,
            radius=radius,
            value=1.0,
        )

    mask_arr = np.array(mask, copy=False)
    xmap_arr = np.array(xmap, copy=False)

    xmap_arr[:, :, :] = xmap_arr[:, :, :] * mask_arr[:, :, :]
    return xmap


def get_sample_frame_from_build(autobuild: StructureI) -> SampleFrameI:
    centroid = get_structure_centroid(autobuild)
    mat = np.eye(3)
    return SampleFrame(
        transform_from_arrays(centroid, mat),
        SAMPLE_SIZE
    )


class LitBuildScoring(lt.LightningModule):
    def __init__(self, ):
        super().__init__()
        self.z_encoder = resnet10(num_classes=2, num_input=3, headless=True).float()
        self.fc_corr = nn.Sequential(

            nn.Linear(512, 1),

        )

    def forward(self, z, ):
        z_encoding = self.z_encoder(z)

        score = F.hardtanh(self.fc_corr(z_encoding), min_val=0.0, max_val=10.0) / 10

        return score


class BuildScorer:

    def __init__(self, model):
        self.model = model

    def __call__(self, autobuild: StructureI, zmap: GridI, xmap: GridI, ) -> float:
        # Get the sample frame
        sample_frame = get_sample_frame_from_build(autobuild)

        # Get the xmap sample
        xmap_sample = sample_frame(mask_xmap_ligand(autobuild, xmap), )

        # Get the zmap sample
        zmap_sample = sample_frame(zmap)

        # Get the ligand mask sample
        ligand_mask = get_ligand_mask(autobuild, zmap)
        ligand_mask_sample = sample_frame(ligand_mask)

        # Get the ligand mask
        ligand_mask_sample_bin = np.copy(ligand_mask_sample)
        ligand_mask_sample_bin[ligand_mask_sample_bin <= 0.0] = 0.0
        ligand_mask_sample_bin[ligand_mask_sample_bin > 0.0] = 1.0

        # Run the model
        return self.model(
            torch.from_numpy(
                np.stack(
                    [
                        zmap_sample,
                        xmap_sample * ligand_mask_sample_bin,
                        ligand_mask_sample
                    ]
                )
            )
        )
