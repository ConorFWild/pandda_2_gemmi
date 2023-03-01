import os
import inspect

import torch
import numpy as np
import gemmi

from pandda_gemmi.cnn import resnet
from pandda_gemmi.analyse_interface import *
from pandda_gemmi.cnn import resnet18
from pandda_gemmi.event.event_scoring_inbuilt import EventScoringResult, LigandFittingResult, ConfomerID, ConformerFittingResult, Conformers


def sample_xmap(xmap, transform, sample_array):
    xmap.interpolate_values(sample_array, transform)
    return sample_array


def get_sample_transform_from_event(event: EventInterface,
                                    sample_distance: float,
                                    n: int,
                                    translation: float):
    # Get basic sample grid transform
    initial_transform = gemmi.Transform()
    scale_matrix = np.eye(3) * sample_distance
    initial_transform.mat.fromlist(scale_matrix.tolist())

    # Get sample grid centroid
    sample_grid_centroid = (np.array([n, n, n]) * sample_distance) / 2

    # Get centre grid transform
    centre_grid_transform = gemmi.Transform()
    centre_grid_transform.vec.fromlist([
        -sample_grid_centroid[0],
        -sample_grid_centroid[1],
        -sample_grid_centroid[2],
    ])

    # Event centre transform
    event_centre_transform = gemmi.Transform()
    event_centre_transform.vec.fromlist([x for x in event.cluster.centroid])

    # Apply random translation
    transform = event_centre_transform.combine(
        centre_grid_transform.combine(
            initial_transform
        )
    )

    transform = gemmi.Transform()
    transform.vec.fromlist([
        event.cluster.centroid[j] - sample_grid_centroid[j]
        for j
        in [0,1,2]
    ])
    corner_0_pos = transform.apply(gemmi.Position(0.0, 0.0, 0.0))
    corner_n_pos = transform.apply(gemmi.Position(
        float(n),
        float(n),
        float(n),
    )
    )
    corner_0 = (corner_0_pos.x, corner_0_pos.y, corner_0_pos.z)
    corner_n = (corner_n_pos.x, corner_n_pos.y, corner_n_pos.z)
    average_pos = [c0 + (cn - c0) / 2 for c0, cn in zip(corner_0, corner_n)]
    event_centroid = event.cluster.centroid
    # logger.debug(f"Centroid: {event_centroid}")
    # logger.debug(f"Corners: {corner_0} : {corner_n} : average: {average_pos}")
    # logger.debug(f"Distance from centroid to average: {gemmi.Position(*average_pos).dist(gemmi.Position(*event_centroid))}")

    return transform


def get_model_map(reference: ReferenceInterface, xmap_event):

    structure = reference.dataset.structure.structure
    new_xmap = gemmi.FloatGrid(xmap_event.nu, xmap_event.nv, xmap_event.nw)
    new_xmap.spacegroup = xmap_event.spacegroup
    new_xmap.set_unit_cell(xmap_event.unit_cell)
    for model in structure:
        for chain in model:
            for residue in chain.get_polymer():
                for atom in residue:
                    new_xmap.set_points_around(
                        atom.pos,
                        radius=1,
                        value=1.0,
                    )

    return new_xmap


def get_event_map(reference_xmap_grid, event: EventInterface, model: ModelInterface):
    reference_xmap_grid_array = np.array(reference_xmap_grid, copy=True)

    event_map_reference_grid = gemmi.FloatGrid(*[reference_xmap_grid.nu,
                                                 reference_xmap_grid.nv,
                                                 reference_xmap_grid.nw,
                                                 ]
                                               )
    event_map_reference_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")  # xmap.xmap.spacegroup
    event_map_reference_grid.set_unit_cell(reference_xmap_grid.unit_cell)

    event_map_reference_grid_array = np.array(event_map_reference_grid,
                                              copy=False,
                                              )

    mean_array = model.mean
    event_map_reference_grid_array[:, :, :] = (reference_xmap_grid_array - (event.bdc.bdc * mean_array)) / (
            1 - event.bdc.bdc)

    return event_map_reference_grid


class GetEventScoreCNN(GetEventScoreCNNInterface):
    tag: Literal["cnn"] = "cnn"

    def __call__(self,
                 test_dtag,
                 model_number,
                 processed_dataset,
                 dataset_xmap,
                 zmap,
                 events: EventsInterface,
                 model,
                 grid,
                 dataset_alignment,
                 max_site_distance_cutoff,
                 min_bdc, max_bdc,
                 reference,
                 res, rate,
                 structure_output_folder,
                 event_map_cut=2.0,
                 below_cut_score=0.0,
                 event_density_score=1.0,
                 protein_score=-1.0,
                 protein_event_overlap_score=0.0,
                 event_fit_num_trys=3,
                 debug: Debug = Debug.DEFAULT,
                 ) -> EventScoringResultsInterface:

        # Get the device
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"

        # Load the model
        model = resnet18(num_classes=2, num_input=4)
        cnn_path = Path(os.path.dirname(inspect.getfile(resnet))) / "model.pt"
        model.load_state_dict(torch.load(cnn_path, map_location=dev))

        # Add model to device
        model.to(dev)
        model.eval()

        # Annotate the events
        event_scores = {}
        for event_id, event in events.events.items():

            n = 30
            sample_array = np.zeros((n, n, n), dtype=np.float32)
            sample_transform = get_sample_transform_from_event(
                event,
                0.5,
                n,
                3.5
            )

            event_map = get_event_map(dataset_xmap.xmap, event, model)
            sample_array_event = np.copy(sample_array)
            image_event = sample_xmap(event_map, sample_transform, sample_array_event)

            sample_array_raw = np.copy(sample_array)
            image_raw = sample_xmap(dataset_xmap.xmap, sample_transform, sample_array_raw)

            sample_array_zmap = np.copy(sample_array)
            image_zmap = sample_xmap(zmap.zmap, sample_transform, sample_array_zmap)

            sample_array_model = np.copy(sample_array)
            model_map = get_model_map(reference, event_map)
            image_model = sample_xmap(model_map, sample_transform, sample_array_model)

            image = np.stack([image_event, image_raw, image_zmap, image_model])

            # Transfer to tensor
            image_t = torch.unsqueeze(torch.from_numpy(image), 0)

            # Move tensors to device
            image_c = image_t.to(dev)

            # Run model
            model_annotation = model(image_c)

            # Track score
            model_annotation = model_annotation.to(torch.device("cpu")).detach().numpy()[0][1]
            print(f"{model_number} {test_dtag.dtag} {event_id.event_idx.event_idx} {model_annotation}")
            event_scores[event_id] = EventScoringResult(
                LigandFittingResult(
                    {
                        ConfomerID(0): ConformerFittingResult(
                            model_annotation,
                            None,
                            None
                        )
                    },
                    Conformers(
                        {},
                        "None",
                        None,
                    ),
                ),
            )


        return event_scores
