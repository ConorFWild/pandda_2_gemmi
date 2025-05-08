import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
# import lightning as lt
import pytorch_lightning as lt
from rich import print as rprint


from .interfaces import *
from .base import get_ligand_mask, iterate_atoms, grid_from_template, get_structure_centroid, transform_from_arrays, \
    SampleFrame, _get_ed_mask_float
from .constants import SAMPLE_SIZE, SAMPLE_SPACING
from .resnet import resnet10, _resnet, BasicBlock


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


# class LitBuildScoring(lt.LightningModule):
#     def __init__(self, ):
#         super().__init__()
#         self.z_encoder = resnet10(num_classes=2, num_input=3, headless=True).float()
#         self.fc_corr = nn.Sequential(
#
#             nn.Linear(512, 1),
#
#         )
#
#     def forward(self, z, ):
#         z_encoding = self.z_encoder(z)
#         score = F.hardtanh(self.fc_corr(z_encoding), min_val=0.0, max_val=10.0) / 10
#         return float(score[0][0])

class LitBuildScoring(lt.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.z_encoder = _resnet(
            'resnet10',
            BasicBlock,
            [config['blocks_1'], config['blocks_2'], config['blocks_3'], config['blocks_4'], ],
            False, False,
            num_classes=2,
            num_input=3,
            headless=True,
            drop_rate=config['drop_rate'],
            config=config,
        ).float()

        self.fc_corr = nn.Sequential(
            nn.Linear(config['planes_5'], 1),

        )
        self.train_annotations = []
        self.test_annotations = []

        # self.output = output_dir
        self.lr = config['lr']
        self.wd = config['wd']
        self.batch_size = config['batch_size']
        self.max_pos_atom_mask_radius = config['max_pos_atom_mask_radius']

    def forward(self, z,):
        z_encoding = self.z_encoder(z)
        corr_hat = ((F.hardtanh(self.fc_corr(z_encoding), min_val=-1.0, max_val=1.0) + 1) / 2) #* self.max_pos_atom_mask_radius


        return corr_hat


class BuildScorer:

    def __init__(self, model, config):
        self.model = model.eval().float()
        self.config = config

    def __call__(self, autobuild: StructureI, zmap: GridI, xmap: GridI, ) -> float:
        # Get the sample frame
        sample_frame = get_sample_frame_from_build(
            autobuild,
            self.config['sample_size'],
            self.config['sample_spacing'],
        )

        # Get the xmap sample
        # masked_xmap = mask_xmap_ligand(autobuild, xmap)
        xmap_sample = sample_frame(xmap, scale=False)

        # Get the zmap sample
        zmap_sample = sample_frame(zmap, scale=False)

        # Get the ligand mask sample
        ligand_mask = get_ligand_mask(
            autobuild,
            zmap,
            radius=self.config['max_pos_atom_mask_radius'],
        )
        ligand_mask_sample = sample_frame(ligand_mask, scale=False)
        ligand_mask_sample[ligand_mask_sample < 0.5] = 0.0
        ligand_mask_sample[ligand_mask_sample >= 0.5] = 1.0

        # Run the model
        arr = np.stack(
                    [
                        zmap_sample * ligand_mask_sample,
                        xmap_sample * ligand_mask_sample,
                        ligand_mask_sample
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
        ).detach().numpy(), arr
