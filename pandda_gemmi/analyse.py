# ###########################################
# # Todo
# ###########################################
# 1. [ x ] Check event calculation makes sense
# 2. [ x ] Check zmaps match up after normalisation
# 4. [ x ] Check event table and site table make sense
# 5. [ x ] Make sure PanDDA inspect works
# 8. [ x ] Check dask submission works on diamond

# Maybe
# 11. [ x ] Email Marcin about cutout maps (mapmask or gemmi)
# 7. [ x ] Get autobuilding working

# More optional
# 3. [ ] Tidy up code
# 6. [ ] Make sure works with XCE
# 9. [ ] Complete logging
# 10. [ x ] Speed up sigma calculation - ?Numba?
# 12. [ ] Remove the random constants from code

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
    PanDDAFSModel, Dataset, Datasets, Reference, Resolution,
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
    get_comparators_high_res_random,
    get_comparators_closest_cutoff,
    get_comparators_closest_apo_cutoff,
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
    process_shell_low_mem,
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
        # Processing settings
        local_processing: str = "multiprocessing_spawn",
        local_cpus: int = 12,
        global_processing: str = "serial",
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
        distributed_walltime="5:0:0",
        distributed_watcher=False,
        distributed_slurm_partition=None,
        # Autobuild settings
        autobuild: bool = False,
        autobuild_strategy: str = "rhofit",
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
    pandda_fs_model.build()

    print("Getting multiprocessor")
    try:
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
        elif global_processing == "fs":
            process_global = FSCluster(
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
                slurm_partition=distributed_slurm_partition,
                output_dir=distributed_tmp,
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
                )

            elif autobuild_strategy == "inbuilt":
                autobuild_parametrized = partial(
                    autobuild_inbuilt,
                )

            else:
                raise Exception(f"Autobuild strategy: {autobuild_strategy} is not valid!")

        # Parameterise
        if low_memory:
            process_shell_paramaterised = partial(
                process_shell_low_mem,
                process_local=process_local,
                structure_factors=structure_factors,
                sample_rate=sample_rate,
                contour_level=contour_level,
                cluster_cutoff_distance_multiplier=cluster_cutoff_distance_multiplier,
                min_blob_volume=min_blob_volume,
                min_blob_z_peak=min_blob_z_peak,
                outer_mask=outer_mask,
                inner_mask_symmetry=inner_mask_symmetry,
                max_site_distance_cutoff=max_site_distance_cutoff,
                min_bdc=min_bdc,
                max_bdc=max_bdc,
            )

        else:
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
                max_site_distance_cutoff=max_site_distance_cutoff,
                min_bdc=min_bdc,
                max_bdc=max_bdc,
            )

        ###################################################################
        # # Pre-pandda
        ###################################################################

        # Get datasets
        print("Loading datasets")
        datasets_initial: Datasets = Datasets.from_dir(pandda_fs_model)
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

        datasets_low_res: Datasets = datasets_invalid.remove_low_resolution_datasets(
            low_resolution_completeness)
        pandda_log[constants.LOG_LOW_RES] = [dtag.dtag for dtag in datasets_invalid if dtag not in datasets_low_res]
        validate_paramterized(datasets_invalid, exception=Exception("Too few datasets after filter: low res"))

        datasets_rfree: Datasets = datasets_low_res.remove_bad_rfree(max_rfree)
        validate_paramterized(datasets_invalid, exception=Exception("Too few datasets after filter: rfree"))

        datasets_wilson: Datasets = datasets_rfree.remove_bad_wilson(
            max_wilson_plot_z_score)  # TODO
        validate_paramterized(datasets_invalid, exception=Exception("Too few datasets after filter: wilson"))

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
        validate_paramterized(datasets_invalid, exception=Exception("Too few datasets after filter: structure"))

        print("Removing models with large gaps")
        datasets_gaps: Datasets = datasets_smoother.remove_models_with_large_gaps(reference, )
        for dtag in datasets_gaps:
            if dtag not in datasets_diss_struc.datasets:
                print(f"WARNING: Removed dataset {dtag} due to a large gap")
        validate_paramterized(datasets_invalid, exception=Exception("Too few datasets after filter: structure gaps"))

        print("Removing dissimilar space groups")
        datasets_diss_space: Datasets = datasets_gaps.remove_dissimilar_space_groups(reference)
        validate_paramterized(datasets_invalid, exception=Exception("Too few datasets after filter: space group"))

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

        # Assign comparator set for each dataset
        if comparison_strategy == "closest":
            # Closest datasets after clustering
            raise NotImplementedError()
            comparators: Dict[Dtag, List[Dtag]] = ...

        elif comparison_strategy == "closest_cutoff":
            # Closest datasets after clustering as long as they are not too poor res
            comparators: Dict[Dtag, List[Dtag]] = get_comparators_closest_cutoff(
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
                exclude_local,
            )

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

        elif comparison_strategy == "get_comparators_closest_apo_cutoff":
            if not known_apos:
                known_apos = [dtag for dtag in datasets if pandda_fs_model.processed_datasets[dtag].source_ligand_cif]
            else:
                known_apos = [Dtag(dtag) for dtag in known_apos]
                for known_apo in known_apos:
                    if known_apo not in datasets:
                        raise Exception(
                            f"Human specified known apo {known_apo} of known apos: {known_apos} not in "
                            f"dataset dtags: {list(datasets.keys())}"
                        )
            print(f"Known apos are: {known_apos}")

            pandda_log[constants.LOG_KNOWN_APOS] = [dtag.dtag for dtag in known_apos]

            comparators: Dict[Dtag, List[Dtag]] = get_comparators_closest_apo_cutoff(
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
                known_apos,
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
        time_shells_start = time.time()
        shell_results: List[ShellResult] = process_global(
            [
                partial(
                    process_shell_paramaterised,
                    shell,
                    datasets,
                    alignments,
                    grid,
                    pandda_fs_model,
                    reference,
                )
                for res, shell
                in shells.items()
            ],
        )
        time_shells_finish = time.time()
        print(f"Finished processing shells in: {time_shells_finish - time_shells_start}")
        pandda_log[constants.LOG_SHELLS] = {
            res: shell_result.log
            for res, shell_result
            in zip(shells, shell_results)
            if shell_result
        }

        all_events: Dict[EventId, Event] = {}
        for shell_result in shell_results:
            if shell_result:
                for dtag, dataset_result in shell_result.dataset_results.items():
                    print(type(dataset_result.events))
                    all_events.update(dataset_result.events.events)

        # Add the event maps to the fs
        for event_id, event in all_events.items():
            pandda_fs_model.processed_datasets[event_id.dtag].event_map_files.add_event(event)

        printer.pprint(all_events)

        ###################################################################
        # # Autobuilding
        ###################################################################

        # Autobuild the results if set to
        if autobuild:
            print("Attempting to autobuild!...")

            time_autobuild_start = time.time()
            autobuild_results_list: Dict[EventID, AutobuildResult] = process_global(
                [
                    partial(
                        autobuild_parametrized,
                        datasets[event_id.dtag],
                        all_events[event_id],
                        pandda_fs_model,
                    )
                    for event_id
                    in all_events
                ]
            )

            time_autobuild_finish = time.time()
            pandda_log[constants.LOG_AUTOBUILD_TIME] = time_autobuild_finish - time_autobuild_start

            autobuild_results: Dict[EventID, AutobuildResult] = {
                event_id: autobuild_result
                for event_id, autobuild_result
                in zip(all_events, autobuild_results_list)
            }
            printer.pprint(autobuild_results)

            # Save results
            pandda_log[constants.LOG_AUTOBUILD_COMMANDS] = {}
            for event_id, autobuild_result in autobuild_results.items():
                dtag = str(event_id.dtag.dtag)
                if dtag not in pandda_log[constants.LOG_AUTOBUILD_COMMANDS]:
                    pandda_log[constants.LOG_AUTOBUILD_COMMANDS][dtag] = {}

                event_idx = int(event_id.event_idx.event_idx)

                pandda_log[constants.LOG_AUTOBUILD_COMMANDS][dtag][event_idx] = autobuild_result.command

            # Add the best fragment by scoring method to default model
            pandda_log[constants.LOG_AUTOBUILD_SELECTED_BUILDS] = {}
            pandda_log[constants.LOG_AUTOBUILD_SELECTED_BUILD_SCORES] = {}
            for dtag in datasets:
                print(f"Finding best autobuild for dataset: {dtag}")
                dataset_autobuild_results = {
                    event_id: autobuild_result
                    for event_id, autobuild_result
                    in autobuild_results.items()
                    if dtag == event_id.dtag
                }

                if len(dataset_autobuild_results) == 0:
                    print("\tNo autobuilds for this dataset!")
                    continue

                all_scores = {}
                for event_id, autobuild_result in dataset_autobuild_results.items():
                    for path, score in autobuild_result.scores.items():
                        all_scores[path] = score

                if len(all_scores) == 0:
                    print(f"\tNo autobuilds for this dataset!")
                    continue

                printer.pprint(all_scores)

                # Select fragment build
                selected_fragement_path = max(
                    all_scores,
                    key=lambda _path: all_scores[_path],
                )

                pandda_log[constants.LOG_AUTOBUILD_SELECTED_BUILDS][dtag.dtag] = str(selected_fragement_path)
                pandda_log[constants.LOG_AUTOBUILD_SELECTED_BUILD_SCORES][dtag.dtag] = float(
                    all_scores[selected_fragement_path])

                print(f"Selected fragment path: {selected_fragement_path}")

                # Copy to pandda models
                model_path = str(pandda_fs_model.processed_datasets[dtag].input_pdb)
                pandda_model_path = pandda_fs_model.processed_datasets[
                                        dtag].dataset_models.path / constants.PANDDA_EVENT_MODEL.format(dtag.dtag)
                merged_structure = merge_ligand_into_structure_from_paths(model_path, selected_fragement_path)
                save_pdb_file(merged_structure, pandda_model_path)

        ###################################################################
        # # Rank Events
        ###################################################################
        if rank_method == "size":
            all_events_ranked = rank_events_size(all_events, grid)
        elif rank_method == "size_delta":
            raise NotImplementedError()
            all_events_ranked = rank_events_size_delta()
        elif rank_method == "cnn":
            raise NotImplementedError()
            all_events_ranked = rank_events_cnn()

        elif rank_method == "autobuild":
            if not autobuild:
                raise Exception("Cannot rank on autobuilds if autobuild is not set!")
            else:
                all_events_ranked = rank_events_autobuild(
                    all_events,
                    autobuild_results,
                    datasets,
                    pandda_fs_model,
                )
        else:
            raise Exception(f"Ranking method: {rank_method} is unknown!")

        ###################################################################
        # # Assign Sites
        ###################################################################

        # Get the events and assign sites to them
        all_events_events = Events.from_all_events(all_events_ranked, grid, max_site_distance_cutoff)

        ###################################################################
        # # Output pandda summary information
        ###################################################################

        # Output a csv of the events
        event_table: EventTable = EventTable.from_events(all_events_events)
        event_table.save(pandda_fs_model.analyses.pandda_analyse_events_file)

        # Output site table
        site_table: SiteTable = SiteTable.from_events(all_events_events, max_site_distance_cutoff)
        site_table.save(pandda_fs_model.analyses.pandda_analyse_sites_file)

        time_finish = time.time()
        print(f"PanDDA ran in: {time_finish - time_start}")
        pandda_log[constants.LOG_TIME] = time_finish - time_start

        # Output json log
        printer.pprint(pandda_log)
        save_json_log(
            pandda_log,
            out_dir / constants.PANDDA_LOG_FILE,
        )

    ###################################################################
    # # Handle Exceptions
    ###################################################################

    except Exception as e:
        traceback.print_exc()

        pandda_log[constants.LOG_TRACE] = traceback.format_exc()
        pandda_log[constants.LOG_EXCEPTION] = str(e)

        print(f"Saving PanDDA log to: {out_dir / constants.PANDDA_LOG_FILE}")

        printer.pprint(
            pandda_log
        )

        save_json_log(
            pandda_log,
            out_dir / constants.PANDDA_LOG_FILE,
        )


if __name__ == '__main__':
    fire.Fire(process_pandda)
