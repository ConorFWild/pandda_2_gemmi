class ScoreCNN:
    ...

    def __call__(self, event):

        centroid = np.mean(event.pos_array, axis=0)

        sample_transform = get_sample_transform_from_event(
            centroid,
            0.5,
            n,
            3.5
        )
        time_finish_get_sample_transform = time.time()
        print(f"Got sample transform in: {time_finish_get_sample_transform - time_begin_get_sample_transform}")

        sample_array = np.zeros((n, n, n), dtype=np.float32)

        bdcs = np.linspace(0.0, 0.95, 20).reshape((20, 1, 1, 1))

        time_begin_get_xmap_sample = time.time()
        xmap_sample = sample_xmap(xmap_grid, sample_transform, np.copy(sample_array))
        time_finish_get_xmap_sample = time.time()
        print(f"Got xmap sample in: {time_finish_get_xmap_sample - time_begin_get_xmap_sample}")

        mean_map_sample = sample_xmap(mean_grid, sample_transform, np.copy(sample_array))

        image_events = (xmap_sample[np.newaxis, :] - (bdcs * mean_map_sample[np.newaxis, :])) / (1 - bdcs)
        print(f"Image evnets: {image_events.shape}")

        # event_map = get_event_map(dataset_xmap.xmap, event, model)
        # sample_array_event = np.copy(sample_array)
        # image_event = sample_xmap(event_map, sample_transform, sample_array_event)

        # sample_array_raw = np.copy(sample_array)
        # image_raw = sample_xmap(dataset_xmap.xmap, sample_transform, sample_array_raw)
        image_raw = np.stack([xmap_sample for _j in range(20)])

        sample_array_zmap = np.copy(sample_array)
        zmap_sample = sample_xmap(z_grid, sample_transform, sample_array_zmap)
        image_zmap = np.stack([zmap_sample for _j in range(20)])

        sample_array_model = np.copy(sample_array)

        model_sample = sample_xmap(model_map, sample_transform, sample_array_model)
        image_model = np.stack([model_sample for _j in range(20)])

        image = np.stack([image_events, image_raw, image_zmap, image_model], axis=1)
        print([image.shape, image.dtype])

        # Transfer to tensor
        # image_t = torch.unsqueeze(torch.from_numpy(image), 0)
        image_t = torch.from_numpy(image)

        # Move tensors to device
        image_c = image_t.to(dev)

        # Run model
        time_begin_run_model = time.time()
        model_annotation = cnn(image_c.float())
        time_finish_run_model = time.time()
        print(f"Model ran in: {time_finish_run_model - time_begin_run_model}")

        # Track score
        model_annotations = model_annotation.to(torch.device("cpu")).detach().numpy()
        for _j in range(20):
            bdc = bdcs.flatten()[_j]
            annotation = model_annotations[_j, 1]
            print(f"\t\t{np.round(bdc, 2)} {round(float(annotation), 2)}")

        flat_bdcs = bdcs.flatten()
        max_score_index = np.argmax([annotation for annotation in model_annotations[:, 1]])
        event_scores[cluster_num] = (
            round(float(flat_bdcs[max_score_index]), 2),
            round(float(model_annotations[max_score_index, 1]), 2),
            [
                round(float(centroid[0]), 2),
                round(float(centroid[1]), 2),
                round(float(centroid[2]), 2),
            ]
        )