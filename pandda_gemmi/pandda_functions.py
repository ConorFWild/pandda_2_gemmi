from __future__ import annotations

from typing import *
from functools import partial
import json

import numpy as np
import multiprocessing as mp
import joblib

from sklearn import decomposition
import umap
from bokeh.plotting import ColumnDataSource, figure, output_file, show

from pandda_gemmi.pandda_types import *


def run(func):
    return func()


def process_local_serial(funcs):
    results = []
    for func in funcs:
        results.append(func())

    return results


def process_local_joblib(n_jobs, verbosity, funcs):
    mapper = joblib.Parallel(n_jobs=n_jobs,
                             verbose=verbosity,
                             backend="loky",
                             )

    results = mapper(joblib.delayed(func)() for func in funcs)

    return results


def process_local_multiprocessing(funcs, n_jobs=12, method="forkserver"):
    if method == "forkserver":
        try:
            mp.set_start_method("forkserver")
        except Exception as e:
            print(e)

    elif method == "spawn":
        try:
            mp.set_start_method("spawn")
        except Exception as e:
            print(e)

    else:
        raise Exception(
            f"Method {method} is not a valid multiprocessing start method: try spawn (stable) or forkserver (fast)")

    with mp.Pool(n_jobs) as pool:
        results = pool.map(run, funcs)

    return results


def process_local_dask(funcs, client=None):
    processes = [client.submit(func) for func in funcs]
    results = client.gather(processes)
    return results


def process_shell_dask(funcs):
    from dask.distributed import worker_client

    with worker_client() as client:
        # Multiprocess
        processes = [client.submit(func) for func in funcs]
        results = client.gather(processes)
    return results


def process_global_serial(funcs):
    results = []
    for func in funcs:
        results.append(func())

    return results


def get_dask_client(scheduler="SGE",
                    num_workers=10,
                    queue=None,
                    project=None,
                    cores_per_worker=12,
                    distributed_mem_per_core=10,
                    resource_spec="",
                    walltime="1:00:00",
                    ):
    import dask
    from dask.distributed import Client
    from dask_jobqueue import HTCondorCluster, PBSCluster, SGECluster, SLURMCluster

    dask.config.set({'distributed.worker.daemon': False})

    schedulers = ["HTCONDOR", "PBS", "SGE", "SLURM"]
    if scheduler not in schedulers:
        raise Exception(f"Supported schedulers are: {schedulers}")

    if scheduler == "HTCONDOR":
        cluster = HTCondorCluster(
            queue=queue,
            project=project,
            cores=cores_per_worker,
            memory=f"{distributed_mem_per_core * cores_per_worker}G",
            resource_spec=resource_spec,
            walltime=walltime,
            processes=1,
            nanny=False,
        )

    elif scheduler == "PBS":
        cluster = PBSCluster(
            queue=queue,
            project=project,
            cores=cores_per_worker,
            memory=f"{distributed_mem_per_core * cores_per_worker}G",
            resource_spec=resource_spec,
            walltime=walltime,
            processes=1,
            nanny=False,
        )

    elif scheduler == "SGE":
        extra = [f"-pe smp {cores_per_worker}", "-V"]
        cluster = SGECluster(
            queue=queue,
            project=project,
            cores=cores_per_worker,
            memory=f"{distributed_mem_per_core * cores_per_worker}G",
            resource_spec=resource_spec,
            walltime=walltime,
            processes=1,
            nanny=False,
            job_extra=extra,
        )

    elif scheduler == "SLURM":
        cluster = SLURMCluster(
            queue=queue,
            project=project,
            cores=cores_per_worker,
            memory=f"{distributed_mem_per_core * cores_per_worker}G",
            resource_spec=resource_spec,
            walltime=walltime,
            processes=1,
            nanny=False,
        )

    else:
        raise Exception("Something has gone wrong process_global_dask")

    # Scale the cluster up to the number of workers
    cluster.scale(jobs=num_workers)

    # Launch the client
    client = Client(cluster)
    return client


def process_global_dask(
        funcs,
        client=None,
):
    # Multiprocess
    processes = [client.submit(func) for func in funcs]
    results = client.gather(processes)

    return results


def get_comparators_high_res_random(
        datasets: Dict[Dtag, Dataset],
        comparison_min_comparators,
        comparison_max_comparators,
):
    dtag_list = [dtag for dtag in datasets]

    dtags_by_res = list(
        sorted(
            dtag_list,
            key=lambda dtag: datasets[dtag].reflections.resolution().resolution,
        )
    )

    highest_res_datasets = dtags_by_res[:comparison_min_comparators + 1]
    highest_res_datasets_max = max(
        [datasets[dtag].reflections.resolution().resolution for dtag in highest_res_datasets])

    comparators = {}
    for dtag in dtag_list:
        current_res = datasets[dtag].reflections.resolution().resolution

        truncation_res = max(current_res, highest_res_datasets_max)

        truncated_datasets = [dtag for dtag in dtag_list if
                              datasets[dtag].reflections.resolution().resolution < truncation_res]

        comparators[dtag] = list(
            np.random.choice(
                truncated_datasets,
                size=comparison_min_comparators,
                replace=False,
            )
        )

    return comparators


def get_distance_matrix(samples: MutableMapping[str, np.ndarray]) -> np.ndarray:
    # Make a pairwise matrix
    correlation_matrix = np.zeros((len(samples), len(samples)))

    for x, reference_sample in enumerate(samples.values()):

        reference_sample_mean = np.mean(reference_sample)
        reference_sample_demeaned = reference_sample - reference_sample_mean
        reference_sample_denominator = np.sqrt(np.sum(np.square(reference_sample_demeaned)))

        for y, sample in enumerate(samples.values()):
            sample_mean = np.mean(sample)
            sample_demeaned = sample - sample_mean
            sample_denominator = np.sqrt(np.sum(np.square(sample_demeaned)))

            nominator = np.sum(reference_sample_demeaned * sample_demeaned)
            denominator = sample_denominator * reference_sample_denominator

            correlation = nominator / denominator

            correlation_matrix[x, y] = correlation

    correlation_matrix = np.nan_to_num(correlation_matrix)

    # distance_matrix = np.ones(correlation_matrix.shape) - correlation_matrix

    for j in range(correlation_matrix.shape[0]):
        correlation_matrix[j, j] = 1.0

    return correlation_matrix


def embed_umap(distance_matrix):
    pca = decomposition.PCA(n_components=min(distance_matrix.shape[0], 50))
    reducer = umap.UMAP()
    transform = pca.fit_transform(distance_matrix)
    transform = reducer.fit_transform(transform)
    return transform


def bokeh_scatter_plot(embedding, labels, know_apos, plot_file):
    output_file(str(plot_file))

    source = ColumnDataSource(
        data=dict(
            x=embedding[:, 0].tolist(),
            y=embedding[:, 1].tolist(),
            dtag=labels,
            apo=["green" if label in know_apos else "pink" for label in labels]
        ))

    TOOLTIPS = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
        ("dtag", "@dtag"),
        ("apo", "@apo"),
    ]

    p = figure(plot_width=1200, plot_height=1200, tooltips=TOOLTIPS,
               title="Mouse over the dots",
               )

    p.circle('x', 'y', size=15, source=source, color="apo")

    show(p)


def save_plot_pca_umap_bokeh(dataset_connectivity_matrix, labels, known_apos, plot_file):
    embedding = embed_umap(dataset_connectivity_matrix)
    bokeh_scatter_plot(embedding, labels, known_apos, plot_file)


def get_comparators_closest_cutoff(
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
):
    dtag_list = [dtag for dtag in datasets]
    dtag_array = np.array(dtag_list)

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
    start = time.time()
    load_xmap_paramaterised = partial(
        Xmap.from_unaligned_dataset_c,
        grid=grid,
        structure_factors=structure_factors,
        sample_rate=sample_rate,
    )

    results = process_local(
        [
            partial(
                load_xmap_paramaterised,
                shell_truncated_datasets[key],
                alignments[key],
            )
            for key
            in shell_truncated_datasets
        ]
    )

    # Get the maps as arrays
    xmaps = {dtag: xmap.to_array()
             for dtag, xmap
             in zip(datasets, results)
             }

    finish = time.time()
    print(f"Mapped in {finish - start}")

    # Get the correlation distance between maps
    correlation_matrix = get_distance_matrix(xmaps)

    # Save a bokeh plot
    save_plot_pca_umap_bokeh(correlation_matrix,
                             labels,
                             known_apos,
                             out_dir / f"pca_umap.html")

    # Get the comparators: for each dataset rank all comparators, then go along accepting or rejecting them
    # Based on whether they are within the res cutoff
    comparators = {}
    for j, dtag in enumerate(dtag_list):
        print(f"Finding closest for dtag: {dtag}")
        current_res = datasets[dtag].reflections.resolution().resolution

        # Get dtags ordered by distance
        row = correlation_matrix[j, :].flatten()
        print(f"\tRow is: {row}")
        closest_dtags_indexes = np.flip(np.argsort(row))
        closest_dtags = np.take_along_axis(dtag_array, closest_dtags_indexes, axis=0)
        print(f"\tClosest dtags are: {closest_dtags}")
        print(f"\tdistances are: {np.take_along_axis(row, closest_dtags_indexes, axis=0)}")

        # Decide the res upper bound
        truncation_res = max(current_res + resolution_cutoff, highest_res_datasets_max)
        print(f"\tTrucation res is: {truncation_res}")

        # Go down the list of closes datasets seeing if they fall within truncation res and adding them to comparators
        # if so

        potential_comparator_dtags = []
        for potential_comparator_dtag in closest_dtags:

            if datasets[dtag].reflections.resolution().resolution < truncation_res:
                potential_comparator_dtags.append(potential_comparator_dtag)
            else:
                continue

            # of enough accuulated, continue
            if len(potential_comparator_dtags) > comparison_min_comparators:
                comparators[dtag] = potential_comparator_dtags
                break

    return comparators


def get_shells(
        datasets: Dict[Dtag, Dataset],
        comparators: Dict[Dtag, List[Dtag]],
        min_characterisation_datasets,
        max_shell_datasets,
        high_res_increment,
):
    # For each dataset + set of comparators, include all of these to be loaded in the set of the shell of their highest
    # Common reoslution

    # Get the dictionary of resolutions for convenience
    resolutions = {dtag: datasets[dtag].reflections.resolution().resolution for dtag in datasets}

    # Get the shells: start with the highest res dataset and count up in increments of high_res_increment to the
    # Lowest res dataset
    reses = np.arange(min(resolutions.values()), max(resolutions.values()), high_res_increment)
    shells_test = {res: set() for res in reses}
    shells_train = {res: {} for res in reses}

    # Iterate over comparators, getting the resolution range, the lowest res in it, and then including all
    # in the set of the first shell of sufficiently low res

    for dtag, comparison_dtags in comparators.items():
        low_res = max([resolutions[comparison_dtag] for comparison_dtag in comparison_dtags])

        # Find the first shell whose res is higher
        for res in reses:
            if res > low_res:
                shells_test[res] = shells_test[res].union({dtag, })
                shells_train[res][dtag] = set(comparison_dtags)

                # Make sure they only appear in one shell
                break

    # Create shells
    shells = {}
    for j, res in enumerate(reses):

        # Collect a set of all dtags
        all_dtags = set()

        # Add all the test dtags
        for dtag in shells_test[res]:
            all_dtags = all_dtags.union({dtag, })

        # Add all the train dtags
        for test_dtag, train_dtags in shells_train[res].items():
            all_dtags = all_dtags.union(train_dtags)

        # Create the shell
        shell = Shell(
            shells_test[res],
            shells_train[res],
            all_dtags,
        )
        shells[res] = shell

    # Delete any shells that are empty
    shells_to_delete = []
    for res in reses:
        if len(shells_test[res]) == 0 or len(shells_train[res]) == 0:
            shells_to_delete.append(res)

    for res in shells_to_delete:
        del shells[res]

    return shells


def truncate(datasets: Dict[Dtag, Dataset], resolution: Resolution, structure_factors: StructureFactors):
    new_datasets_resolution = {}

    # Truncate by common resolution
    for dtag in datasets:
        truncated_dataset = datasets[dtag].truncate_resolution(resolution, )

        new_datasets_resolution[dtag] = truncated_dataset

    dataset_resolution_truncated = Datasets(new_datasets_resolution)

    # Get common set of reflections
    common_reflections = dataset_resolution_truncated.common_reflections(structure_factors)

    # truncate on reflections
    new_datasets_reflections = {}
    for dtag in dataset_resolution_truncated:
        reflections = dataset_resolution_truncated[dtag].reflections.reflections
        reflections_array = np.array(reflections)
        print(f"{dtag}")
        print(f"{reflections_array.shape}")

        truncated_dataset = dataset_resolution_truncated[dtag].truncate_reflections(common_reflections,
                                                                                    )
        reflections = truncated_dataset.reflections.reflections
        reflections_array = np.array(reflections)
        print(f"{dtag}: {reflections_array.shape}")

        new_datasets_reflections[dtag] = truncated_dataset

    return new_datasets_reflections


def validate_strategy_num_datasets(datasets, min_characterisation_datasets=30):
    if len(datasets) < min_characterisation_datasets:
        return False
    else:
        return True


def validate(datasets: Dict[Dtag, Dataset], strategy=None, exception=None):
    if not strategy(datasets):
        print(datasets)
        raise exception
