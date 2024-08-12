import numpy as np
import gemmi

from .interfaces import *
from .base import transform_from_arrays, SampleFrame, grid_from_template, get_ligand_mask
from .constants import SAMPLE_SIZE


def get_sample_frame_from_event(event: EventI) -> SampleFrameI:
    return SampleFrame(transform_from_arrays(event.centroid, np.eye(3)), SAMPLE_SIZE)


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
    def __init__(self, output_dir):
        super().__init__()
        self.resnet = resnet10(num_classes=2, num_input=1, headless=True).float()
        self.z_encoder = resnet10(num_classes=2, num_input=2, headless=True).float()
        self.x_encoder = SimpleConvolutionalEncoder(input_layers=1)
        self.mol_encoder = resnet10(num_classes=2, num_input=1, headless=True).float()
        self.mol_decoder = SimpleConvolutionalDecoder()
        self.x_decoder = SimpleConvolutionalDecoder(input_layers=512)
        self.z_decoder = SimpleConvolutionalDecoder(input_layers=512)
        self.mol_to_weight = nn.Linear(512, 512)
        self.bn = nn.BatchNorm1d(512)
        self.fc = nn.Sequential(
            nn.Linear(512,2),
        )
        self.train_annotations = []
        self.test_annotations = []

        self.output = output_dir

    def forward(self, x, z, m, d):
        mol_encoding = self.mol_encoder(m)
        z_encoding = self.z_encoder(z)

        full_encoding = z_encoding * F.hardtanh(mol_encoding, min_val=-1.0, max_val=1.0)

        score = F.softmax(self.fc(full_encoding))

        return score

class EventScorer:

    def __init__(self, model):
        self.model = model

    def __call__(self, event: EventI, ligand_conformation: StructureI, zmap: GridI, xmap: GridI) -> float:
        # Get the sample frame
        sample_frame = get_sample_frame_from_event(event)

        # Get the xmap sample
        x, y, z = event.centroid
        xmap_sample = sample_frame(mask_xmap_radial(xmap, x, y, z), )

        # Get the zmap sample
        zmap_sample = sample_frame(zmap)

        # Get the ligand mask sample
        ligand_mask = get_ligand_mask(ligand_conformation, zmap)
        ligand_mask_sample = sample_frame(ligand_mask)

        # Get the ligand mask
        ligand_mask_sample_bin = np.copy(ligand_mask_sample)
        ligand_mask_sample_bin[ligand_mask_sample_bin <= 0.0] = 0.0
        ligand_mask_sample_bin[ligand_mask_sample_bin > 0.0] = 1.0

        # Run the model
        return self.model(
            np.stack(
                [
                    zmap_sample,
                    xmap_sample,

                ]
            ),
            np.stack(
                [
                    ligand_mask_sample
                ]
            )
        )
