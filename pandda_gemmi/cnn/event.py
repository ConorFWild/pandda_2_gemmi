import time

import numpy as np
from skimage.segmentation import expand_labels
import gemmi
import torch
from torch import nn
from torch.nn import functional as F
# import lightning as lt
import pytorch_lightning as lt

from .interfaces import *
from .base import transform_from_arrays, SampleFrame, grid_from_template, get_ligand_mask, get_structure_array, copy_map, _get_ed_mask_float
from .constants import SAMPLE_SIZE, SAMPLE_SPACING
from .resnet import resnet10, _resnet, BasicBlock


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


# class LitEventScoring(lt.LightningModule):
#     def __init__(self, ):
#         super().__init__()
#         self.z_encoder = resnet10(num_classes=2, num_input=2, headless=True).float()
#         self.mol_encoder = resnet10(num_classes=2, num_input=1, headless=True).float()
#         self.fc = nn.Sequential(
#             nn.Linear(512, 2),
#         )
#
#     def forward(self, z, m, ):
#         mol_encoding = self.mol_encoder(m)
#         z_encoding = self.z_encoder(z)
#
#         full_encoding = z_encoding * F.hardtanh(mol_encoding, min_val=-1.0, max_val=1.0)
#
#         score = F.softmax(self.fc(full_encoding))
#
#         print(score)
#
#         return float(score[0][1])


class LitEventScoring(lt.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.z_encoder = _resnet(
            'resnet10',
            BasicBlock,
            [config['blocks_1'], config['blocks_2'], config['blocks_3'], config['blocks_4'], ],
            False, False,
            num_classes=3, num_input=2, headless=True, drop_rate=config['drop_rate'], config=config).float()
        self.mol_encoder = _resnet(
            'resnet10',
            BasicBlock,
            [config['blocks_1'], config['blocks_2'], config['blocks_3'], config['blocks_4'], ],
            False, False,
            num_classes=2, num_input=1, headless=True, drop_rate=config['drop_rate'], config=config).float()

        self.fc = nn.Sequential(
            nn.Linear(config['planes_5'] , 3),

        )
        self.train_annotations = []
        self.test_annotations = []

        # self.output = output_dir
        self.lr = config['lr']
        self.wd = config['wd']
        self.batch_size = config['batch_size']

    def forward(self, z, m, ):
        mol_encoding = self.mol_encoder(m)
        z_encoding = self.z_encoder(z)
        full_encoding = z_encoding * mol_encoding
        score = F.softmax(self.fc(full_encoding))

        return score


class EventScorer:

    def __init__(self, model, config):
        self.model = model.eval().float()
        self.config = config

    def __call__(self, event: EventI, ligand_conformation: StructureI, zmap: GridI, xmap: GridI) -> float:
        # Get the sample frame
        sample_frame = get_sample_frame_from_event(
            event,
            self.config['sample_size'],
            self.config['sample_spacing']
        )

        # Cut the xmap
        # mask = _get_ed_mask_float()

        # Get the xmap sample
        xmap_sample = sample_frame(xmap, scale=False)

        # Get the zmap sample
        zmap_sample = sample_frame(zmap, scale=False)

        # Get the ligand mask sample
        ligand_mask = get_ligand_mask(ligand_conformation, zmap)
        ligand_mask_sample = sample_frame(ligand_mask, scale=False)

        #
        high_z_mask = (zmap_sample > self.config['z_cutoff']).astype(int)
        high_z_mask_expanded = expand_labels(high_z_mask, distance=self.config['z_mask_radius'] / 0.5)
        high_z_mask_expanded[high_z_mask_expanded != 1] = 0

        # Run the model
        map_array = np.stack(
                    [
                        zmap_sample * high_z_mask,
                        xmap_sample * high_z_mask,
                    ],
            dtype=np.float32
                )[np.newaxis,:]
        mol_array = np.stack(
                [
                    ligand_mask_sample
                ],
            dtype=np.float32
            )[np.newaxis,:]

        return self.model(
            torch.from_numpy(map_array),
            torch.from_numpy(mol_array)
        ).detach().numpy(), map_array, mol_array
