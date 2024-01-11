import numpy as np


def filter_characterization_sets(
        comparator_datasets,
        characterization_sets,
        dmaps,
        dataset_dmap_array,
        reference_frame,
        outlier_model,
        process_all=False
):
    characterization_set_masks = {}
    model_scores = {}

    for model_number, characterization_set in characterization_sets.items():

        # Get the characterization set dmaps
        characterization_set_mask_list = []
        for _dtag in comparator_datasets:
            if _dtag in characterization_set:
                characterization_set_mask_list.append(True)
            else:
                characterization_set_mask_list.append(False)
        characterization_set_mask = np.array(characterization_set_mask_list)
        characterization_set_masks[model_number] = characterization_set_mask

        characterization_set_dmaps_array = dmaps[characterization_set_masks[model_number], :]
        mean, std, z = outlier_model(
            dataset_dmap_array,
            characterization_set_dmaps_array
        )
        inner_mask_zmap = z[reference_frame.mask.indicies_sparse_inner_atomic]
        percentage_z_2 = float(np.sum(np.abs(inner_mask_zmap) > 2)) / inner_mask_zmap.size
        # print(f"Model number: {model_number}: z > 2: {percentage_z_2}")
        # print(f"Model number: {model_number}: {np.min(std)} {np.mean(std)}  {np.max(std)} {np.std(std)}")
        # print(f"Model number: {np.quantile(z, (0.8, 0.85, 0.9, 0.95))}")

        # mean_grid = reference_frame.unmask(SparseDMap(mean))
        # mean_grid_array = np.array(mean_grid, copy=False)
        # print(mean_grid_array.shape)
        # print(np.nonzero(mean_grid_array == 0))
        # mask_array = np.zeros(mean_grid_array.shape)
        # mask_array[reference_frame.mask.indicies] = 1
        # non_zero = np.nonzero((mean_grid_array == 0) & (mask_array == 1))
        # for j in range(non_zero[0].size):
        #     print(f"{non_zero[0][j]} : {non_zero[1][j]} : {non_zero[2][j]}")
        # print(np.)

        model_scores[model_number] = percentage_z_2

    models_to_process = []
    if process_all:
        for model_number in sorted(model_scores, key=lambda _model_number: model_scores[_model_number]):
            if model_scores[model_number] < 0.2:
                models_to_process.append(model_number)
    else:
        _l = 0
        for model_number in sorted(model_scores, key=lambda _model_number: model_scores[_model_number]):
            if (_l < 3) or (model_scores[model_number] < 0.2):
                models_to_process.append(model_number)
                _l = _l + 1

    return models_to_process, model_scores, characterization_set_masks
