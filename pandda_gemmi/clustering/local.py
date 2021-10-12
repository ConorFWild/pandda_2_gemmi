
def get_comparators_closest_cluster_neighbours(
        datasets: Dict[Dtag, Dataset],
        alignments,
        grid,
        comparison_min_comparators,
        comparison_max_comparators,
        structure_factors,
        sample_rate,
        resolution_cutoff,
        pandda_fs_model: PanDDAFSModel,
        process_local,
        batch=False,
        cluster_selection="close"
):
    dtag_list = [dtag for dtag in datasets]
    dtag_array = np.array(dtag_list)
    dtag_to_index = {dtag: j for j, dtag in enumerate(dtag_list)}

    dtags_by_res = list(
        sorted(
            dtag_list,
            key=lambda dtag: datasets[dtag].reflections.resolution().resolution,
        )
    )

    highest_res_datasets = dtags_by_res[:comparison_min_comparators + 1]
    highest_res_datasets_max = max(
        [datasets[dtag].reflections.resolution().resolution for dtag in highest_res_datasets])

    # Load the xmaps
    print("Truncating datasets...")
    shell_truncated_datasets: Datasets = truncate(
        datasets,
        resolution=Resolution(highest_res_datasets_max),
        structure_factors=structure_factors,
    )

    # Generate aligned xmaps
    print("Loading xmaps")

    load_xmap_paramaterised = partial(
        from_unaligned_dataset_c_flat,
        grid=grid,
        structure_factors=structure_factors,
        sample_rate=sample_rate,
    )

    # Get reduced array
    total_sample_size = len(shell_truncated_datasets)
    print(f"Total sample size = {total_sample_size}")
    batch_size = min(90, total_sample_size)
    print(f"Batch size is: {batch_size}")
    num_batches = (total_sample_size // batch_size) + 1
    print(f"Num batches is: {num_batches}")
    # batches = [
    #     np.arange(x*batch_size, min((x+1)*batch_size, total_sample_size))
    #     for x
    #     in range(0, num_batches)]
    tmp_batches = {}
    j = 1
    while True:
        print(f"\tJ is: {j}")
        new_batches = np.array_split(np.arange(total_sample_size), j)
        print(f"\t\tlen of new batches is {len(new_batches)}")
        tmp_batches[j] = new_batches
        j = j + 1

        if any(len(batch) < batch_size for batch in new_batches):
            batches = tmp_batches[j - 2]
            break
        else:
            print("\t\tAll batches larger than batch size, trying smaller split!")
            continue
    print(f"Batches are:")
    print(batches)

    from sklearn.decomposition import PCA, IncrementalPCA
    ipca = IncrementalPCA(n_components=min(200, batch_size))

    print("Fitting!")
    for batch in batches:
        print(f"\tLoading dtags: {dtag_array[batch]}")
        start = time.time()
        results = process_local(
            [
                partial(
                    load_xmap_paramaterised,
                    shell_truncated_datasets[key],
                    alignments[key],
                )
                for key
                in dtag_array[batch]
            ]
        )
        print("Got xmaps!")

        # Get the maps as arrays
        print("Getting xmaps as arrays")
        xmaps = {dtag: xmap
                 for dtag, xmap
                 in zip(dtag_list, results)
                 }

        finish = time.time()
        print(f"Mapped in {finish - start}")

        # Get pca
        xmap_array = np.vstack([xmap for xmap in xmaps.values()])
        ipca.partial_fit(xmap_array)

    # Transform
    print(f"Transforming!")
    transformed_arrays = []
    for batch in batches:
        print(f"\tTransforming dtags: {dtag_array[batch]}")
        start = time.time()
        results = process_local(
            [
                partial(
                    load_xmap_paramaterised,
                    shell_truncated_datasets[key],
                    alignments[key],
                )
                for key
                in dtag_array[batch]
            ]
        )
        print("Got xmaps!")

        # Get the maps as arrays
        print("Getting xmaps as arrays")
        xmaps = {dtag: xmap
                 for dtag, xmap
                 in zip(dtag_list, results)
                 }

        finish = time.time()
        print(f"Mapped in {finish - start}")

        # Get pca
        xmap_array = np.vstack([xmap for xmap in xmaps.values()])
        transformed_arrays.append(ipca.transform(xmap_array))

    reduced_array = np.vstack(transformed_arrays)

    print(f"Reduced array shape: {reduced_array.shape}")

    # clusterer = hdbscan.HDBSCAN(
    #     min_cluster_size=30,
    #     min_samples=1,
    #     cluster_selection_method="leaf",
    # )
    # clusterer.fit(reduced_array)
    # labels = clusterer.labels_
    # print(f"Labels are: {labels}")
    # probabilities = clusterer.probabilities_
    #
    # # Plot cluster results
    # fig, ax = plt.subplots()
    #
    # clusterer.condensed_tree_.plot(
    #     select_clusters=True,
    #     axis=ax,
    # )
    #
    # fig.savefig(str(pandda_fs_model.pandda_dir / f"hdbscan_condensed_tree.png"))
    # fig.clear()
    # plt.close(fig)
    #
    # # Plot cluster results
    # fig, ax = plt.subplots()
    #
    # clusterer.single_linkage_tree_.plot(
    #     axis=ax,
    # )
    #
    # fig.savefig(str(pandda_fs_model.pandda_dir / f"hdbscan_single_linkage_tree.png"))
    # fig.clear()
    # plt.close(fig)

    # # Plot cluster results
    # fig, ax = plt.subplots()
    #
    # clusterer.minimum_spanning_tree_.plot(
    #     axis=ax,
    # )
    #
    # fig.savefig(str(pandda_fs_model.pandda_dir / f"hdbscan_minimum_spanning_tree.png"))
    # fig.clear()
    # plt.close(fig)

    linkage = get_linkage_from_observations(reduced_array)

    save_dendrogram_plot(linkage,
                         [_dtag.dtag for _dtag in dtag_list],
                         str(pandda_fs_model.pandda_dir / f"dendrogram.png"),
                         # threshold=0.3,
                         )

    rootnode, nodelist = spc.hierarchy.to_tree(linkage, rd=True)

    def recurse_node(node, min_samples):

        _clusters = []
        # if node.count > min_samples:
        if node.left.count >= min_samples:
            left_clusters = recurse_node(node.left, min_samples)
            for left_cluster in left_clusters:
                _clusters.append(left_cluster)

        if node.right.count >= min_samples:
            right_clusters = recurse_node(node.right, min_samples)
            for right_cluster in right_clusters:
                _clusters.append(right_cluster)

        if node.count >= min_samples:
            if (node.right.count < min_samples) and (node.left.count < min_samples):
                _clusters.append(node.pre_order(lambda x: x.id))

        return _clusters

    clusters = recurse_node(rootnode, 30)
    print(clusters)

    clusters_dict = {}
    dtag_to_cluster = {}
    for j, cluster in enumerate(clusters):
        clusters_dict[j] = dtag_array[np.array(cluster)]
        for dtag in clusters_dict[j]:
            dtag_to_cluster[dtag] = j
    print(clusters_dict)

    save_dendrogram_plot(linkage,
                         [
                             f"{_dtag.dtag}_{dtag_to_cluster[_dtag]}" if _dtag in dtag_to_cluster else _dtag.dtag
                             for _dtag
                             in dtag_list
                         ],
                         str(pandda_fs_model.pandda_dir / f"dendrogram_with_clusters.png"),
                         # threshold=0.3,
                         )

    #
    #
    # # Get the cores of each cluster
    # cluster_cores = {}
    # for n in np.unique(labels):
    #     if n != -1:
    #         indexes = np.arange(len(labels))
    #         cluster_member_mask = labels == n
    #         cluster_member_indexes = np.nonzero(cluster_member_mask)
    #         cluster_member_values = probabilities[cluster_member_mask]
    #         cluster_members_sorted_indexes = np.argsort(cluster_member_values)
    #
    #         if np.sum(cluster_member_indexes) >= 30:
    #             cluster_cores[n] = cluster_member_indexes[cluster_member_mask][cluster_members_sorted_indexes][:30]
    #
    #         else:
    #             print(f"There were less than 30 members of the cluster!")
    #
    # print(f"Cluster cores are:")
    # print(cluster_cores)

    # Save a bokeh plot
    labels = [dtag.dtag for dtag in dtag_list]
    # known_apos = [dtag.dtag for dtag, dataset in datasets.items() if any(dtag in x for x in cluster_cores.values())]
    known_apos = []
    for cluster_num, cluster_dtags in clusters_dict.items():
        print(f"\tCluster {cluster_num} dtags are {cluster_dtags}")
        for cluster_core_dtag in cluster_dtags:
            known_apos.append(cluster_core_dtag.dtag)

    print(f"Labels are: {labels}")
    print(f"Known apos are: {known_apos}")

    save_plot_pca_umap_bokeh(
        reduced_array,
        labels,
        known_apos,
        pandda_fs_model.pandda_dir / f"pca_umap.html",
    )

    #
    cophenetic_matrix = spsp.distance.squareform(spc.hierarchy.cophenet(linkage))
    dtag_distance_to_cluster = {}
    for _dtag in dtag_list:
        dtag_index = dtag_to_index[_dtag]
        dtag_distance_to_cluster[_dtag] = {}
        dtag_coord = reduced_array[dtag_index, :]
        for cluster, cluster_dtags in clusters_dict.items():
            cluster_indexes = np.array([dtag_to_index[_cluster_dtag] for _cluster_dtag in cluster_dtags])
            cluster_coords = reduced_array[cluster_indexes, :]

            cluster_squared_vectors = np.sqrt(np.sum(np.square(cluster_coords - dtag_coord), axis=1))

            median_squared_distance = np.median(cluster_squared_vectors)

            dtag_distance_to_cluster[_dtag][cluster] = median_squared_distance

    cluster_widths = {}
    for cluster, cluster_dtags in clusters_dict.items():
        cluster_indexes = np.array([dtag_to_index[cluster_dtag] for cluster_dtag in cluster_dtags])
        cluster_coords = reduced_array[cluster_indexes, :]
        cluster_median = np.median(cluster_coords, axis=0).reshape((1, cluster_coords.shape[1]))
        cluster_median_deviation = np.median(np.sqrt(np.sum(np.square(cluster_coords - cluster_median), axis=1)))
        cluster_widths[cluster] = cluster_median_deviation

    print(f"Cluster median absolute deviation is: {cluster_widths}")

    # Get the centermost cluster
    cluster_medians = {}
    for cluster, cluster_dtags in clusters_dict.items():
        cluster_indexes = np.array([dtag_to_index[cluster_dtag] for cluster_dtag in cluster_dtags])
        cluster_coords = reduced_array[cluster_indexes, :]
        cluster_median = np.median(cluster_coords, axis=0).reshape((1, cluster_coords.shape[1]))
        cluster_medians[cluster] = cluster_median
    print(f"Cluster medians are: {cluster_medians}")

    median_of_medians = np.median(np.vstack([x for x in cluster_medians.values()]), axis=0).reshape(1,
                                                                                                    cluster_coords.shape[
                                                                                                        1])
    print(f"Global median of clusters is: {median_of_medians}")

    centermost_cluster = min(
        cluster_medians,
        key=lambda _cluster_num: np.sqrt(np.sum(np.square((median_of_medians - cluster_medians[_cluster_num])))),
    )
    print(f"Centermost cluster is: {centermost_cluster}")

    # Get the comparators: for each dataset, get cluster with closest median distance
    comparators = {}
    for j, dtag in enumerate(dtag_list):
        print(f"Finding closest for dtag: {dtag}")
        current_res = datasets[dtag].reflections.resolution().resolution

        # Get dtags ordered by distance
        row = cophenetic_matrix[j, :].flatten()
        print(f"\tRow is: {row}")
        # closest_dtags_indexes = np.flip(np.argsort(row))
        # cluster_distances = {}
        # for cluster_num, cluster_dtags in clusters_dict.items():
        #     distances = row[np.array([dtag_to_index[_dtag] for _dtag in cluster_dtags])]
        #     median_distance = np.median(distances)
        #     print(f"\t\tMedian distance to cluster {cluster_num} is: {median_distance}")
        #     cluster_distances[cluster_num] = median_distance

        # print(f"\tClosest dtags are: {closest_dtags}")
        # print(f"\tdistances are: {np.take_along_axis(row, closest_dtags_indexes, axis=0)}")

        cluster_distances = dtag_distance_to_cluster[dtag]

        if cluster_selection == "close":

            print(cluster_distances)
            closest_cluster = min(cluster_distances, key=lambda x: cluster_distances[x])
            # print(f"\tClosest cluster is: {closest_cluster}")
            # closest_cluster_dtags = clusters_dict[closest_cluster]
            # print(f"\tClosest cluster dtags ate: {closest_cluster_dtags}")

        elif cluster_selection == "center":
            closest_cluster = centermost_cluster
            # closest_cluster_dtags = clusters_dict[closest_cluster]

        elif cluster_selection == "far":
            closest_cluster = max(cluster_distances, key=lambda x: cluster_distances[x])

        elif cluster_selection == "next":
            cluster_distances_sorted = list(sorted(cluster_distances, key=lambda x: cluster_distances[x]))
            if len(cluster_distances) < 2:
                closest_cluster = cluster_distances_sorted[0]
            else:
                closest_cluster = cluster_distances_sorted[1]

        print(f"\tClosest cluster is: {closest_cluster}")
        closest_cluster_dtags = clusters_dict[closest_cluster]
        print(f"\tClosest cluster dtags ate: {closest_cluster_dtags}")

        distances_to_cluster = {_dtag: dtag_distance_to_cluster[_dtag][closest_cluster]
                                for _dtag
                                in dtag_distance_to_cluster}
        dtags_by_distance_to_cluster = [x for x in sorted(distances_to_cluster, key=lambda y: distances_to_cluster[y])]
        print(f"Distances to cluster: {distances_to_cluster}")
        print(f"Dtags by distance to cluster: {dtags_by_distance_to_cluster}")

        # Decide the res upper bound
        truncation_res = max(current_res + resolution_cutoff, highest_res_datasets_max)
        print(f"\tTrucation res is: {truncation_res}")

        # Go down the list of closes datasets seeing if they fall within truncation res and adding them to comparators
        # if so
        potential_comparator_dtags = []
        for j, potential_comparator_dtag in enumerate(dtags_by_distance_to_cluster):

            if datasets[dtag].reflections.resolution().resolution < truncation_res:
                potential_comparator_dtags.append(potential_comparator_dtag)
            else:
                continue

            # of enough accuulated, continue
            if len(potential_comparator_dtags) > comparison_min_comparators:
                comparators[dtag] = potential_comparator_dtags
                break

        if len(potential_comparator_dtags) < comparison_min_comparators:
            raise Exception(
                (
                    f"Dtag {dtag} has too few comparators: "
                    f"only {len(potential_comparator_dtags)}:"
                    f" {potential_comparator_dtags}"
                )
            )

    return comparators
