
from __future__ import annotations

# Base python
import traceback
from typing import Dict, Optional, List, Tuple
import time
from pathlib import Path
import pprint
from functools import partial
import multiprocessing as mp
import inspect

# Scientific python libraries
import fire
import numpy as np
import shutil

## Custom Imports
from pandda_gemmi.logs import (
    summarise_grid, summarise_event, summarise_structure, summarise_mtz, summarise_array, save_json_log,
    summarise_datasets
)
from pandda_gemmi.pandda_types import (
    PanDDAFSModel, ShellDirs, Dataset, Datasets, Reference, Resolution,
    Grid, Alignments, Shell, Xmap, Xmaps, Zmap,
    XmapArray, Model, Dtag, Zmaps, Clustering, Clusterings,
    EventID, Event, Events, SiteTable, EventTable,
    StructureFactors, Xmap,
    DatasetResult, ShellResult,
    AutobuildResult
)
from pandda_gemmi import constants
from pandda_gemmi.pandda_functions import (
    process_local_serial,
    process_local_joblib,
    process_local_multiprocessing,
    get_dask_client,
    process_global_serial,
    process_global_dask,
    get_shells,
get_comparators_high_res,
    get_comparators_high_res_random,
    get_comparators_closest_cutoff,
    get_comparators_closest_apo_cutoff,
    get_clusters_linkage,
    get_clusters_nn,
    get_comparators_closest_cluster,
    truncate,
    validate_strategy_num_datasets,
    validate,
    get_common_structure_factors,
)
from pandda_gemmi.ranking import (
    rank_events_size,
    rank_events_autobuild,
)
from pandda_gemmi.autobuild import (
    autobuild_rhofit,
    merge_ligand_into_structure_from_paths,
    save_pdb_file,
)
from pandda_gemmi.distribution.fscluster import FSCluster

from pandda_gemmi.processing import (
    process_shell,
    # process_shell_low_mem,
)

printer = pprint.PrettyPrinter()


def process_pandda(
        # IO options
        data_dirs: str,
        out_dir: str,
        pdb_regex: str = "*.pdb",
        mtz_regex: str = "*.mtz",
        ligand_dir_name: str = "compound",
        ligand_cif_regex: str = "*.cif",
        ligand_pdb_regex: str = "*.pdb",
        ligand_smiles_regex: str = "*.smiles",
        statmaps: bool = False,
        # PROCESS MANAGEMENT,
        low_memory: bool = False,
        # Dataset selection options
        ground_state_datasets: Optional[List[str]] = None,
        exclude_from_z_map_analysis: Optional[List[str]] = None,
        exclude_from_characterisation: Optional[List[str]] = None,
        only_datasets: Optional[List[str]] = None,
        ignore_datasets: Optional[List[str]] = None,
        # Shell determination options
        dynamic_res_limits: bool = True,
        high_res_upper_limit: float = 1.0,
        high_res_lower_limit: float = 4.0,
        high_res_increment: float = 0.05,
        max_shell_datasets: int = 60,
        min_characterisation_datasets: int = 15,
        # Diffraction data options
        structure_factors: Optional[Tuple[str, str]] = None,
        all_data_are_valid_values: bool = True,
        low_resolution_completeness: float = 4.0,
        sample_rate: float = 3.0,
        # Dataset filtering options
        max_rmsd_to_reference: float = 1.5,
        max_rfree: float = 0.4,
        max_wilson_plot_z_score=1.5,
        same_space_group_only: bool = False,
        similar_models_only: bool = False,
        # Map options
        resolution_factor: float = 0.25,
        grid_spacing: float = 0.5,
        padding: float = 3.0,
        density_scaling: bool = True,
        outer_mask: float = 8.0,
        inner_mask: float = 2.0,
        inner_mask_symmetry: float = 2.0,
        # ZMap clustering options
        contour_level: float = 2.5,
        negative_values: bool = False,
        min_blob_volume: float = 10.0,
        min_blob_z_peak: float = 3.0,
        clustering_cutoff: float = 1.5,
        cluster_cutoff_distance_multiplier: float = 1.0,
        # Site finding options
        max_site_distance_cutoff=1.732,
        # BDC options
        min_bdc: float = 0.0,
        max_bdc: float = 0.95,
        increment: float = 0.05,
        output_multiplier: float = 2.0,
        # Comparater set finding options
        comparison_strategy: str = "closest_cutoff",
        comparison_res_cutoff: float = 0.5,
        comparison_min_comparators: int = 30,
        comparison_max_comparators: int = 30,
        known_apos: Optional[List[str]] = None,
        exclude_local: int = 5,
        cluster_selection: str ="close",
        # Processing settings
        local_processing: str = "multiprocessing_spawn",
        local_cpus: int = 12,
        global_processing: str = "serial",
        memory_availability: str = "high",
        job_params_file: Optional[str] = None,
        # Distributed settings
        distributed_scheduler: str = "SGE",
        distributed_queue: str = "medium.q",
        distributed_project: str = "labxchem",
        distributed_num_workers: int = 12,
        distributed_cores_per_worker: int = 12,
        distributed_mem_per_core: int = 10,
        distributed_resource_spec: str = "m_mem_free=10G",
        distributed_tmp: str = "/tmp",
        job_extra: str = ["--exclusive", ],
        distributed_walltime="30:00:00",
        distributed_watcher=False,
        distributed_slurm_partition=None,
        # Autobuild settings
        autobuild: bool = False,
        autobuild_strategy: str = "rhofit",
        rhofit_coord: bool = False,
        cif_strategy: str = "elbow",
        # Ranking settings
        rank_method: str = "size",
        # Debug settings
        debug: bool = True,
):
    ###################################################################
    # # Configuration
    ###################################################################
    time_start = time.time()
    print("Getting config")

    # Process args
    initial_arg_values = inspect.getargvalues(inspect.currentframe())
    printer.pprint(initial_arg_values)
    data_dirs = Path(data_dirs)
    out_dir = Path(out_dir)
    distributed_tmp = Path(distributed_tmp)

    if structure_factors:
        structure_factors = StructureFactors(f=structure_factors[0], phi=structure_factors[1])

    print("Initialising log...")
    pandda_log: Dict = {}
    pandda_log[constants.LOG_START] = time.time()
    pandda_log[constants.LOG_ARGUMENTS] = initial_arg_values

    print("FSmodel building")
    pandda_fs_model: PanDDAFSModel = PanDDAFSModel.from_dir(
        data_dirs,
        out_dir,
        pdb_regex,
        mtz_regex,
        ligand_dir_name,
        ligand_cif_regex,
        ligand_pdb_regex,
        ligand_smiles_regex,
    )

    print("Getting multiprocessor")
    # Get global processor
    if global_processing == "serial":
        process_global = process_global_serial
    elif global_processing == "distributed":
        client = get_dask_client(
            scheduler=distributed_scheduler,
            num_workers=distributed_num_workers,
            queue=distributed_queue,
            project=distributed_project,
            cores_per_worker=local_cpus,
            distributed_mem_per_core=distributed_mem_per_core,
            resource_spec=distributed_resource_spec,
            job_extra=job_extra,
            walltime=distributed_walltime,
            watcher=distributed_watcher,
        )
        process_global = partial(
            process_global_dask,
            client=client,
            tmp_dir=distributed_tmp
        )
    else:
        raise Exception()

    # Get local processor
    if local_processing == "serial":
        raise NotImplementedError()
        process_local = ...
    elif local_processing == "joblib":
        process_local = partial(process_local_joblib, n_jobs=local_cpus, verbose=0)
    elif local_processing == "multiprocessing_forkserver":
        mp.set_start_method("forkserver")
        process_local = partial(process_local_multiprocessing, n_jobs=local_cpus, method="forkserver")
    elif local_processing == "multiprocessing_spawn":
        mp.set_start_method("spawn")
        process_local = partial(process_local_multiprocessing, n_jobs=local_cpus, method="spawn")
    else:
        raise Exception()

    # Set up autobuilding
    if autobuild:

        if autobuild_strategy == "rhofit":
            ana_pdbmaps_path = shutil.which("ana_pdbmaps")
            rhofit_path = shutil.which("rhofit")
            pandda_rhofit_path = shutil.which("pandda_rhofit.sh")

            if not ana_pdbmaps_path:
                raise Exception("PanDDA Rhofit requires ana_pdbmaps to be in path!")
            if not rhofit_path:
                raise Exception("PanDDA Rhofit requires rhofit to be in path!")
            if not pandda_rhofit_path:
                raise Exception("PanDDA Rhofit requires pandda_rhofit.sh to be in path!")

            autobuild_parametrized = partial(
                autobuild_rhofit,
                cif_strategy=cif_strategy,
                rhofit_coord=rhofit_coord,
            )

        elif autobuild_strategy == "inbuilt":
            autobuild_parametrized = partial(
                autobuild_inbuilt,
            )

        else:
            raise Exception(f"Autobuild strategy: {autobuild_strategy} is not valid!")


    ###################################################################
    # # Pre-pandda
    ###################################################################

    # Get datasets
    print("Loading datasets")
    datasets_initial: Datasets = Datasets.from_dir(pandda_fs_model,)
    print(f"\tThere are initially: {len(datasets_initial)} datasets")

    # If structure factors not given, check if any common ones are available
    if not structure_factors:
        structure_factors = get_common_structure_factors(datasets_initial)
        # If still no structure factors
        if not structure_factors:
            raise Exception(
                "No common structure factors found in mtzs. Please manually provide the labels with the --structure_factors option.")

    # Make dataset validator
    validation_strategy = partial(validate_strategy_num_datasets,
                                  min_characterisation_datasets=min_characterisation_datasets,
                                  )
    validate_paramterized = partial(validate, strategy=validation_strategy)

    # Initial filters
    print("Filtering invalid datasaets")
    datasets_invalid: Datasets = datasets_initial.remove_invalid_structure_factor_datasets(
        structure_factors)
    pandda_log[constants.LOG_INVALID] = [dtag.dtag for dtag in datasets_initial if dtag not in datasets_invalid]
    validate_paramterized(datasets_invalid, exception=Exception("Too few datasets after filter: invalid"))

    datasets_truncated_columns = datasets_invalid.drop_columns(structure_factors)

    datasets_low_res: Datasets = datasets_truncated_columns.remove_low_resolution_datasets(
        low_resolution_completeness)
    pandda_log[constants.LOG_LOW_RES] = [dtag.dtag for dtag in datasets_truncated_columns if dtag not in datasets_low_res]
    validate_paramterized(datasets_low_res, exception=Exception("Too few datasets after filter: low res"))

    datasets_rfree: Datasets = datasets_low_res.remove_bad_rfree(max_rfree)
    validate_paramterized(datasets_rfree, exception=Exception("Too few datasets after filter: rfree"))

    datasets_wilson: Datasets = datasets_rfree.remove_bad_wilson(
        max_wilson_plot_z_score)  # TODO
    validate_paramterized(datasets_wilson, exception=Exception("Too few datasets after filter: wilson"))

    # Select refernce
    print("Getting reference")
    reference: Reference = Reference.from_datasets(datasets_wilson)

    # Post-reference filters
    print("smoothing")
    start = time.time()
    datasets_smoother: Datasets = datasets_wilson.smooth_datasets(
        reference,
        structure_factors=structure_factors,
        mapper=process_local,
    )
    finish = time.time()
    print(f"Smoothed in {finish - start}")

    print("Removing dissimilar models")
    datasets_diss_struc: Datasets = datasets_smoother.remove_dissimilar_models(
        reference,
        max_rmsd_to_reference,
    )
    validate_paramterized(datasets_diss_struc, exception=Exception("Too few datasets after filter: structure"))

    print("Removing models with large gaps")
    datasets_gaps: Datasets = datasets_smoother.remove_models_with_large_gaps(reference, )
    for dtag in datasets_gaps:
        if dtag not in datasets_diss_struc.datasets:
            print(f"WARNING: Removed dataset {dtag} due to a large gap")
    validate_paramterized(datasets_gaps, exception=Exception("Too few datasets after filter: structure gaps"))

    print("Removing dissimilar space groups")
    datasets_diss_space: Datasets = datasets_gaps.remove_dissimilar_space_groups(reference)
    validate_paramterized(datasets_diss_space, exception=Exception("Too few datasets after filter: space group"))

    datasets = {dtag: datasets_diss_space[dtag] for dtag in datasets_diss_space}
    pandda_log[constants.LOG_DATASETS] = summarise_datasets(datasets, pandda_fs_model)

    # Grid
    print("Getting grid")
    grid: Grid = Grid.from_reference(reference,
                                     outer_mask,
                                     inner_mask_symmetry,
                                     sample_rate=sample_rate,
                                     )

    if debug > 1:
        print("Summarising protein mask")
        summarise_grid(grid.partitioning.protein_mask)
        print("Summarising symmetry mask")
        summarise_grid(grid.partitioning.symmetry_mask)
        print("Summarising total mask")
        summarise_grid(grid.partitioning.total_mask)

    print("Getting alignments")
    alignments: Alignments = Alignments.from_datasets(
        reference,
        datasets,
    )

    ###################################################################
    # # Assign comparison datasets
    ###################################################################

    print("Assigning comparison datasets...")
    get_clusters = get_clusters_nn
    # Closest datasets after clustering as long as they are not too poor res
    comparators: Dict[Dtag, List[Dtag]] = get_comparators_closest_cluster(
        datasets,
        alignments,
        grid,
        comparison_min_comparators,
        comparison_max_comparators,
        structure_factors,
        sample_rate,
        comparison_res_cutoff,
        pandda_fs_model,
        process_local,
        get_clusters,
        cluster_selection=cluster_selection,
    )



if __name__ == '__main__':
    fire.Fire(process_pandda)
