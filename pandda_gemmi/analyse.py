from __future__ import annotations

import os
from pandda_gemmi import constants
from pandda_gemmi.constants import PANDDA_LOG_FILE
from typing import Dict, Optional, List, Tuple, Union
import time
import psutil
import pickle
from shlex import split
from pprint import PrettyPrinter
from pathlib import Path
import pprint
from functools import partial
import multiprocessing as mp

printer = pprint.PrettyPrinter()

import fire
import numpy as np
import gemmi
import joblib
from joblib.externals.loky import set_loky_pickler

from pandda_gemmi.config import Config
from pandda_gemmi import logs
from pandda_gemmi.pandda_types import (JoblibMapper, PanDDAFSModel, Dataset, Datasets, Reference, Resolution,
                                       Grid, Alignments, Shell, Xmaps,
                                       XmapArray, Model, Dtag, Zmaps, Clustering, Clusterings,
                                       Events, SiteTable, EventTable,
                                       JoblibMapper, Event, SequenceAlignment, StructureFactors, Xmap,
                                       )
from pandda_gemmi import validators
from pandda_gemmi import constants
from pandda_gemmi.pandda_functions import (
    process_local_joblib,
    process_local_multiprocessing,
    process_global_serial,
    get_shells,
    get_comparators_high_res_random,
    truncate,
    validate_strategy_num_datasets,
    validate,
)

set_loky_pickler('pickle')


# ########
# debug
# #########
def summarise_grid(grid: gemmi.FloatGrid):
    grid_array = np.array(grid, copy=False)
    print((
        f"Grid size: {grid.nu} {grid.nv} {grid.nw} \n"
        f"Grid spacegroup: {grid.spacegroup} \n"
        f"Grid unit cell: {grid.unit_cell} \n"
        f"Grid max: {np.max(grid_array)} \n"
        f"Grid min: {np.min(grid_array)} \n"
        f"Grid mean: {np.mean(grid_array)} \n"
    ))


def summarise_mtz(mtz: gemmi.Mtz):
    mtz_array = np.array(mtz, copy=False)
    print(
        (
            f"Mtz shape: {mtz_array.shape} \n"
            f"Mtz spacegroup: {mtz.spacegroup} \n"
        )
    )


def summarise_structure(structure: gemmi.Structure):
    num_models: int = 0
    num_chains: int = 0
    num_residues: int = 0
    num_atoms: int = 0

    for model in structure:
        num_models += 1
        for chain in model:
            num_chains += 1
            for residue in chain:
                num_residues += 1
                for atom in residue:
                    num_atoms += 1

    print(
        (
            f"Num models: {num_models}"
            f"Num chains: {num_chains}"
            f"Num residues: {num_residues}"
            f"Num atoms: {num_atoms}"
        )
    )


# def summarise_event(event: Event):
#     print(
#     (
#         f"Event system: {event.system}\n"
#         f"Event dtag: {event.dtag}\n"
#         f"Event xyz: {event.x} {event.y} {event.z}\n"
#     )
# )


# Define how to process a shell
def process_shell(
        shell: Shell,
        datasets: Dict[Dtag, Dataset],
        alignments,
        grid,
        pandda_fs_model,
        reference,
        process_local,
        structure_factors: StructureFactors,
        sample_rate: float,
        contour_level,
        cluster_cutoff_distance_multiplier,
        min_blob_volume,
        min_blob_z_peak,
        outer_mask,
        inner_mask_symmetry,
):
    print(f"Working on shell: {shell}")
    # pandda_log.shells_log[shell.number] = logs.ShellLog.from_shell(shell)

    # Seperate out test and train datasets
    shell_datasets: Dict[Dtag, Dataset] = {
        dtag: dataset
        for dtag, dataset
        in datasets.items()
        if dtag in shell.all_dtags
    }  # datasets.from_dtags(shell.all_dtags)

    print("Truncating datasets")
    shell_truncated_datasets: Datasets = truncate(
        shell_datasets,
        resolution=Resolution(min([datasets[dtag].reflections.resolution().resolution for dtag in shell.all_dtags])),
        structure_factors=structure_factors,
    )

    # Assign datasets
    # shell_train_datasets: Datasets = shell_truncated_datasets.from_dtags(shell.train_dtags)
    # shell_test_datasets: Datasets = shell_truncated_datasets.from_dtags(shell.test_dtags)

    # Generate aligned xmaps
    print("Loading xmaps")
    start = time.time()
    # xmaps = Xmaps.from_aligned_datasets_c(
    #     shell_truncated_datasets,
    #     alignments,
    #     grid,
    #     structure_factors,
    #     sample_rate=sample_rate,
    #     mapper=process_local,
    # )  # n x (grid size) with total_mask > 0

    load_xmap_paramaterised = partial(
        Xmap.from_unaligned_dataset_c,
        grid=grid,
        structure_factors=structure_factors,
        sample_rate=sample_rate,
    )

    results = process_local(
        partial(
            load_xmap_paramaterised,
            datasets[key],
            alignments[key],
        )
        for key
        in datasets
    )

    xmaps = {dtag: xmap
             for dtag, xmap
             in zip(datasets, results)
             }

    finish = time.time()
    print(f"Mapped in {finish - start}")

    # Seperate out test and train maps
    shell_test_xmaps: Dict[Dtag, Xmap] = {dtag: xmap for dtag, xmap in xmaps.items() if dtag in shell.test_dtags}

    # Get arrays for model
    print("Getting xmap arrays...")
    masked_xmap_array: XmapArray = XmapArray.from_xmaps(
        xmaps,
        grid,
    )  # Size of n x (total mask  > 0)
    masked_train_xmap_array: XmapArray = masked_xmap_array.from_dtags(shell.train_dtags)
    masked_test_xmap_array: XmapArray = masked_xmap_array.from_dtags(shell.test_dtags)

    # Determine the parameters of the model to find outlying electron density
    print("Fitting model")
    mean_array: np.ndarray = Model.mean_from_xmap_array(masked_train_xmap_array,
                                                        )  # Size of grid.partitioning.total_mask > 0

    print("fitting sigma i")
    sigma_is: Dict[Dtag, float] = Model.sigma_is_from_xmap_array(masked_xmap_array,
                                                                 mean_array,
                                                                 1.5,
                                                                 )  # size of n
    print(sigma_is)
    # pandda_log.shells_log[shell.number].sigma_is = {dtag.dtag: sigma_i
    #                                                 for dtag, sigma_i
    #                                                 in sigma_is.items()}

    print("fitting sigma s m")
    sigma_s_m: np.ndarray = Model.sigma_sms_from_xmaps(masked_train_xmap_array,
                                                       mean_array,
                                                       sigma_is,
                                                       )  # size of total_mask > 0
    print(np.min(sigma_s_m))

    model: Model = Model.from_mean_is_sms(
        mean_array,
        sigma_is,
        sigma_s_m,
        grid,
    )

    # model.save_maps(pandda_fs_model.pandda_dir, shell, grid)

    # Calculate z maps
    print("Getting zmaps")
    zmaps: Zmaps = Zmaps.from_xmaps(
        model=model,
        xmaps=shell_test_xmaps,
    )

    # if config.debug > 1:
    # print("saving zmaps")
    for dtag in zmaps:
        if dtag.dtag in constants.MISSES:
            zmap = zmaps[dtag]
            pandda_fs_model.processed_datasets.processed_datasets[dtag].z_map_file.save(zmap)

            xmap = xmaps[dtag]
            path = pandda_fs_model.processed_datasets.processed_datasets[dtag].path / "xmap.ccp4"
            xmap.save(path)

    # Get the clustered electron desnity outliers
    print("clusting")
    # clusterings: Clusterings = Clusterings.from_Zmaps(
    #     zmaps,
    #     reference,
    #     grid,
    #     config.params.masks.contour_level,
    #     cluster_cutoff_distance_multiplier=config.params.blob_finding.cluster_cutoff_distance_multiplier,
    #     mapper=process_local,
    # )
    cluster_paramaterised = partial(
        Clustering.from_zmap,
        reference=reference,
        grid=grid,
        contour_level=contour_level,
        cluster_cutoff_distance_multiplier=cluster_cutoff_distance_multiplier,
    )

    clusterings = process_local(
        [
            partial(cluster_paramaterised, zmaps[dtag],)
            for dtag
            in zmaps
        ]
    )
    clusterings = Clustering({dtag: clustering for dtag, clustering in zip(zmaps, clusterings)})

    # pandda_log.shells_log[shell.number].initial_clusters = logs.ClusteringsLog.from_clusters(
    #     clusterings, grid)

    # Filter out small clusters
    clusterings_large: Clusterings = clusterings.filter_size(grid,
                                                             min_blob_volume,
                                                             )
    # pandda_log.shells_log[shell.number].large_clusters = logs.ClusteringsLog.from_clusters(
    #     clusterings_large, grid)

    # Filter out weak clusters (low peak z score)
    clusterings_peaked: Clusterings = clusterings_large.filter_peak(grid,
                                                                    min_blob_z_peak)
    # pandda_log.shells_log[shell.number].peaked_clusters = logs.ClusteringsLog.from_clusters(
    #     clusterings_peaked, grid)

    clusterings_merged = clusterings_peaked.merge_clusters()
    # pandda_log.shells_log[shell.number].clusterings_merged = logs.ClusteringsLog.from_clusters(
    #     clusterings_merged, grid)

    # Calculate the shell events
    print("getting events")
    print(f"\tGot {len(clusterings_merged.clusters)} clusters")
    events: Events = Events.from_clusters(
        clusterings_merged,
        model,
        xmaps,
        grid,
        1.732,  # TODO: make this a variable ;')
        process_local,
    )
    # pandda_log.shells_log[shell.number].events = logs.EventsLog.from_events(events, grid)
    # print(pandda_log.shells_log[shell.number].events)

    # Save the event maps!
    print("print events")
    events.save_event_maps(shell_truncated_datasets,
                           alignments,
                           xmaps,
                           model,
                           pandda_fs_model,
                           grid,
                           structure_factors,
                           outer_mask,
                           inner_mask_symmetry,
                           mapper=process_local,
                           )

    for event_id in events:
        # Save zmaps
        # zmap = zmaps[event_id.dtag]
        # pandda_fs_model.processed_datasets.processed_datasets[event_id.dtag].z_map_file.save(zmap)

        # xmap = xmaps[event_id.dtag]
        # path = pandda_fs_model.processed_datasets.processed_datasets[event_id.dtag].path / "xmap.ccp4"
        # xmap.save(path)

        # Add events
        all_events[event_id] = events[event_id]

    return all_events


def main(
        data_dirs: str,
        out_dir: str,
        pdb_regex: str = "*.pdb",
        mtz_regex: str = "*.mtz",
        ligand_cif_regex: str = "*.cif",
        ligand_pdb_regex: str = "*.pdb",
        ground_state_datasets: Optional[List[str]] = None,
        exclude_from_z_map_analysis: Optional[List[str]] = None,
        exclude_from_characterisation: Optional[List[str]] = None,
        only_datasets: Optional[List[str]] = None,
        ignore_datasets: Optional[List[str]] = None,
        dynamic_res_limits: bool = True,
        high_res_upper_limit: float = 1.0,
        high_res_lower_limit: float = 4.0,
        high_res_increment: float = 0.05,
        max_shell_datasets: int = 60,
        min_characterisation_datasets: int = 15,
        structure_factors: Optional[Tuple[str, str]] = None,
        all_data_are_valid_values: bool = True,
        low_resolution_completeness: float = 4.0,
        sample_rate: float = 3.0,
        max_rmsd_to_reference: float = 1.5,
        max_rfree: float = 0.4,
        max_wilson_plot_z_score=1.5,
        same_space_group_only: bool = False,
        similar_models_only: bool = False,
        resolution_factor: float = 0.25,
        grid_spacing: float = 0.5,
        padding: float = 3.0,
        density_scaling: bool = True,
        outer_mask: float = 6.0,
        inner_mask: float = 3.0,
        inner_mask_symmetry: float = 3.0,
        contour_level: float = 2.5,
        negative_values: bool = False,
        min_blob_volume: float = 8.0,
        min_blob_z_peak: float = 3.0,
        clustering_cutoff: float = 1.5,
        cluster_cutoff_distance_multiplier: float = 1.0,
        min_bdc: float = 0.0,
        max_bdc: float = 1.0,
        increment: float = 0.05,
        output_multiplier: float = 2.0,
        local_cpus: int = 12,
        job_params_file: Optional[str] = None,
        comparison_strategy: str = "high_res_random",
        comparison_res_cutoff: float = 0.25,
        comparison_min_comparators: int = 15,
        comparison_max_comparators: int = 30,
        local_processing: str = "multiprocessing",
        global_processing: str = "serial",
        debug: bool = True,
):
    ###################################################################
    # # Configuration
    ###################################################################
    print("Getting config")
    # config: Config = Config.from_args()
    # Process args
    data_dirs = Path(data_dirs)
    out_dir = Path(out_dir)
    structure_factors = StructureFactors(f=structure_factors[0], phi=structure_factors[1])
    print(structure_factors)

    print("Initialising log...")
    pandda_log: logs.LogData = logs.LogData.initialise()
    # pandda_log.config = config

    print("FSmodel building")
    pandda_fs_model: PanDDAFSModel = PanDDAFSModel.from_dir(data_dirs,
                                                            out_dir,
                                                            pdb_regex,
                                                            mtz_regex,
                                                            ligand_cif_regex,
                                                            ligand_pdb_regex,
                                                            )
    pandda_fs_model.build()
    pandda_log.fs_log = logs.FSLog.from_pandda_fs_model(pandda_fs_model)

    print("Getting multiprocessor")

    # Get local processor
    if local_processing == "serial":
        raise NotImplementedError()
        process_local = ...
    elif local_processing == "joblib":
        process_local = partial(process_local_joblib, n_jobs=local_cpus, verbose=0)
    elif local_processing == "multiprocessing":
        mp.set_start_method("forkserver")
        process_local = partial(process_local_multiprocessing, n_jobs=local_cpus)
    else:
        raise Exception()

    # Get global processor
    if global_processing == "serial":
        process_global = process_global_serial
    elif global_processing == "cluster":
        raise NotImplementedError()
        process_global = ...
    else:
        raise Exception()

    # Parameterise
    process_shell_paramaterised = partial(
        process_shell,
        process_local=process_local,
        structure_factors=structure_factors,
        sample_rate=sample_rate,
        contour_level=contour_level,
        cluster_cutoff_distance_multiplier=cluster_cutoff_distance_multiplier,
        min_blob_volume=min_blob_volume,
        min_blob_z_peak=min_blob_z_peak,
        outer_mask=outer_mask,
        inner_mask_symmetry=inner_mask_symmetry,
    )

    ###################################################################
    # # Pre-pandda
    ###################################################################

    # Get datasets
    print("Loading datasets")
    datasets_initial: Datasets = Datasets.from_dir(pandda_fs_model)
    pandda_log.preprocessing_log.initial_datasets_log = logs.InitialDatasetLog.from_initial_datasets(datasets_initial)
    print(f"\tThere are initially: {len(datasets_initial)} datasets")

    # datasets_initial: Datasets = datasets_initial.trunate_num_datasets(100)

    # Make dataset validator
    validation_strategy = validate_strategy_num_datasets(min_characterisation_datasets)
    validate_paramterized = partial(validate, strategy=validation_strategy)

    # Initial filters
    print("Filtering invalid datasaets")
    datasets_invalid: Datasets = datasets_initial.remove_invalid_structure_factor_datasets(
        structure_factors)
    pandda_log.preprocessing_log.invalid_datasets_log = logs.InvalidDatasetLog.from_datasets(datasets_initial,
                                                                                             datasets_invalid)
    validate_paramterized(datasets_invalid, exception=Exception("Too few datasets after filter: invalid"))

    datasets_low_res: Datasets = datasets_invalid.remove_low_resolution_datasets(
        low_resolution_completeness)
    pandda_log.preprocessing_log.low_res_datasets_log = logs.InvalidDatasetLog.from_datasets(datasets_invalid,
                                                                                             datasets_low_res)
    dataset_validator.validate(datasets_low_res, constants.STAGE_FILTER_LOW_RESOLUTION)

    datasets_rfree: Datasets = datasets_low_res.remove_bad_rfree(max_rfree)
    pandda_log.preprocessing_log.rfree_datasets_log = logs.RFreeDatasetLog.from_datasets(datasets_low_res,
                                                                                         datasets_rfree)
    dataset_validator.validate(datasets_rfree, constants.STAGE_FILTER_RFREE)

    datasets_wilson: Datasets = datasets_rfree.remove_bad_wilson(
        max_wilson_plot_z_score)  # TODO
    pandda_log.preprocessing_log.wilson_datasets_log = logs.WilsonDatasetLog.from_datasets(datasets_rfree,
                                                                                           datasets_wilson)
    dataset_validator.validate(datasets_wilson, constants.STAGE_FILTER_WILSON)

    # Select refernce
    print("Getting reference")
    reference: Reference = Reference.from_datasets(datasets_wilson)
    pandda_log.reference_log = logs.ReferenceLog.from_reference(reference)

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
    pandda_log.preprocessing_log.smoothing_datasets_log = logs.SmoothingDatasetLog.from_datasets(datasets_smoother)

    print("Removing dissimilar models")
    datasets_diss_struc: Datasets = datasets_smoother.remove_dissimilar_models(
        reference,
        max_rmsd_to_reference,
    )
    pandda_log.preprocessing_log.struc_datasets_log = logs.StrucDatasetLog.from_datasets(datasets_smoother,
                                                                                         datasets_diss_struc)
    dataset_validator.validate(datasets_diss_struc, constants.STAGE_FILTER_STRUCTURE)

    print("Removing models with large gaps")
    datasets_gaps: Datasets = datasets_smoother.remove_models_with_large_gaps(reference, )
    pandda_log.preprocessing_log.struc_datasets_log = logs.StrucDatasetLog.from_datasets(
        datasets_diss_struc,
        datasets_gaps)
    for dtag in datasets_gaps:
        if dtag not in datasets_diss_struc.datasets:
            print(f"WARNING: Removed dataset {dtag} due to a large gap")
    dataset_validator.validate(datasets_gaps, constants.STAGE_FILTER_GAPS)

    print("Removing dissimilar space groups")
    datasets_diss_space: Datasets = datasets_gaps.remove_dissimilar_space_groups(reference)
    pandda_log.preprocessing_log.space_datasets_log = logs.SpaceDatasetLog.from_datasets(
        datasets_gaps,
        datasets_diss_space)
    dataset_validator.validate(datasets_diss_space, constants.STAGE_FILTER_SPACE_GROUP)

    datasets = {dtag: datasets_diss_space[dtag] for dtag in datasets_diss_space}

    # Grid
    print("Getting grid")
    grid: Grid = Grid.from_reference(reference,
                                     outer_mask,
                                     inner_mask_symmetry,
                                     sample_rate=sample_rate,
                                     )
    # grid.partitioning.save_maps(pandda_fs_model.pandda_dir)
    pandda_log.grid_log = logs.GridLog.from_grid(grid)
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
    pandda_log.alignments_log = logs.AlignmentsLog.from_alignments(alignments)

    ###################################################################
    # # Assign comparison datasets
    ###################################################################

    # Assign comparator set for each dataset
    if comparison_strategy == "closest":
        # Closest datasets after clustering
        raise NotImplementedError()
        comparators: Dict[Dtag, List[Dtag]] = ...

    elif comparison_strategy == "closest_cutoff":
        # Closest datasets after clustering as long as they are not too poor res
        raise NotImplementedError()
        comparators: Dict[Dtag, List[Dtag]] = ...

    elif comparison_strategy == "high_res":
        # Almost Old PanDDA strategy: highest res datasets
        raise NotImplementedError()
        comparators: Dict[Dtag, List[Dtag]] = ...

    elif comparison_strategy == "high_res_random":
        # Old pandda strategy: random datasets that are higher resolution
        comparators: Dict[Dtag, List[Dtag]] = get_comparators_high_res_random(
            datasets,
            comparison_min_comparators,
            comparison_max_comparators,
        )

    else:
        raise Exception("Unrecognised comparison strategy")

    print("Comparators are:")
    if debug:
        printer.pprint(comparators)

    ###################################################################
    # # Process shells
    ###################################################################

    # Partition the Analysis into shells in which all datasets are being processed at a similar resolution for the
    # sake of computational efficiency
    shells = get_shells(
        datasets,
        comparators,
        min_characterisation_datasets,
        max_shell_datasets,
        high_res_increment,
    )

    print("Shells are:")
    if debug:
        printer.pprint(shells)

    # Process the shells
    shell_results = process_global(
        [
            lambda: process_shell_paramaterised(
                shell,
                datasets,
                alignments,
                grid,
                pandda_fs_model,
                reference,
            )
            for res, shell in shells.items()
        ],
    )

    # Autobuild the results if set to
    if autobuild_results:
        process_global([lambda result: autobuild(result) for result in shell_results])

    #
    all_events_events = Events.from_all_events(all_events, grid, 1.7)

    # Get the sites and output a csv of them
    site_table: SiteTable = SiteTable.from_events(all_events_events, 1.7)
    site_table.save(pandda_fs_model.analyses.pandda_analyse_sites_file)
    pandda_log.sites_log = logs.SitesLog.from_sites(site_table)

    # Output a csv of the events
    event_table: EventTable = EventTable.from_events(all_events_events)
    event_table.save(pandda_fs_model.analyses.pandda_analyse_events_file)
    pandda_log.events_log = logs.EventsLog.from_events(all_events_events,
                                                       grid,
                                                       )

    pandda_log.save_json(config.output.out_dir / PANDDA_LOG_FILE)


if __name__ == '__main__':
    fire.Fire(main)
