import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import lightning as lt
from rich import print as rprint


from .interfaces import *
from .base import get_ligand_mask, iterate_atoms, grid_from_template, get_structure_centroid, transform_from_arrays, \
    SampleFrame
from .constants import SAMPLE_SIZE, SAMPLE_SPACING
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


def get_sample_frame_from_build(autobuild: StructureI, sample_size, sample_spacing) -> SampleFrameI:
    vec = get_structure_centroid(autobuild) - (0.5*sample_spacing*np.array(sample_size))
    mat = np.eye(3) * sample_spacing
    return SampleFrame(
        transform_from_arrays(vec, mat),
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
        return float(score[0][0])


class BuildScorer:

    def __init__(self, model):
        self.model = model

    def __call__(self, autobuild: StructureI, zmap: GridI, xmap: GridI, ) -> float:
        # Get the sample frame
        sample_frame = get_sample_frame_from_build(autobuild, SAMPLE_SIZE, SAMPLE_SPACING, )

        # Get the xmap sample
        # masked_xmap = mask_xmap_ligand(autobuild, xmap)
        xmap_sample = sample_frame(xmap, scale=True)

        # Get the zmap sample
        zmap_sample = sample_frame(zmap, scale=True)

        # Get the ligand mask sample
        ligand_mask = get_ligand_mask(autobuild, zmap, radius=2.0)
        ligand_mask_sample = sample_frame(ligand_mask, scale=False)

        # Get the ligand mask
        ligand_mask_sample_bin = np.copy(ligand_mask_sample)
        ligand_mask_sample_bin[ligand_mask_sample_bin <= 0.0] = 0.0
        ligand_mask_sample_bin[ligand_mask_sample_bin > 0.0] = 1.0

        # Get the ligand map sample
        ligand_map = get_ligand_mask(autobuild, zmap)
        ligand_map_sample = sample_frame(ligand_map, scale=False)


        # Run the model
        arr = np.stack(
                    [
                        zmap_sample,
                        xmap_sample * ligand_mask_sample_bin,
                        ligand_map_sample
                    ]
                )[np.newaxis,:]

        print(
            f'Build Score Zmap: {round(np.min(arr[0][0]), 3)} {round(np.median(arr[0][0]), 3)} {round(np.max(arr[0][0]), 3)} {round(np.sum(arr[0][0]), 3)}\n'
            f'Build Score xmap: {round(np.min(arr[0][1]), 3)} {round(np.median(arr[0][1]), 3)} {round(np.max(arr[0][1]), 3)} {round(np.sum(arr[0][1]), 3)}\n'
            f'Build Score mask: {round(np.min(arr[0][2]), 3)} {round(np.median(arr[0][2]), 3)} {round(np.max(arr[0][2]), 3)} {round(np.sum(arr[0][2]), 3)}\n'

        )
        return self.model(
            torch.from_numpy(
                arr
            )
        ), arr
