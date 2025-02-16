import inspect
import os
import time

import numpy as np

from rich.console import Console
from rich.panel import Panel
from rich.align import Align
from rich.padding import Padding
from rich.table import Table
from rich.pretty import Pretty
from rich.columns import Columns

from pandda_gemmi.interfaces import *
from pandda_gemmi import constants
from pandda_gemmi.args import PanDDAArgs

import subprocess


def get_git_revision_hash() -> str:
    # return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    f = inspect.getframeinfo(inspect.currentframe()).filename
    path_to_check = Path(os.path.dirname(os.path.abspath(f)))
    p = subprocess.Popen(
        f"cd {path_to_check}; git rev-parse HEAD",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    stdout, stderr = p.communicate()

    return str(stdout.decode('ascii').strip())


def get_git_revision_short_hash() -> str:
    # return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    f = inspect.getframeinfo(inspect.currentframe()).filename
    path_to_check = Path(os.path.dirname(os.path.abspath(f)))
    p = subprocess.Popen(
        f"cd {path_to_check}; git rev-parse --short HEAD",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    stdout, stderr = p.communicate()

    return str(stdout.decode('ascii').strip())


class PanDDAConsole:

    def __init__(self):
        self.console = Console(record=True)

    def start_pandda(self):
        version = constants.VERSION
        git_hash = get_git_revision_short_hash()

        # Print the program title
        printable = self.wrap_title("PanDDA 2")
        self.console.print(printable)

        # Print the version info
        printable = self.indent_text(f"PanDDA Version: {version}")
        self.console.print(printable)

        printable = self.indent_text(f"PanDDA Version GitHub Commit Hash: {git_hash}")
        self.console.print(printable)

    def start_parse_command_line_args(self):
        printable = self.wrap_title("Parsing Command Line Arguments")
        self.console.print(printable)

    def summarise_arguments(self, args):
        self.console.print(args)

    def wrap_title(self, string):
        return Panel(Align.center(f"[bold]{string}"))

    def wrap_subtitle(self, string):
        return self.console.rule(string)

    def start_dependancy_check(self):
        printable = self.wrap_title(constants.CONSOLE_START_DEPENDENCY_CHECK)
        self.console.print(printable)

    def print_failed_dependencies(self, failed_dependency_list):
        printable = self.indent_text("Failed some dependency checks!")
        self.console.print(printable)

        for failed_dependency in failed_dependency_list:

            printable = self.indent_text(f"The options...", indent=8)
            self.console.print(printable)
            for option in failed_dependency.options:
                printable = self.indent_text(f"{option.name} : {option.option_value}", indent=12)
                self.console.print(printable)

            printable = self.indent_text(f"Induce the unmet dependencies...", indent=8)
            self.console.print(printable)
            for dependency in failed_dependency.dependencies:
                printable = self.indent_text(f"{dependency.name} : {dependency.status}", indent=12)
                self.console.print(printable)

        printable = self.indent_text("PanDDA will now exit...")
        self.console.print(printable)

    def print_successful_dependency_check(self):
        printable = self.indent_text("All dependencies are present!")
        self.console.print(printable)

    def start_log(self):
        printable = self.wrap_title(constants.CONSOLE_START_LOG)
        self.console.print(printable)

    def started_log(self, log_path):
        printable = self.indent_text(f"Started log at {log_path}")
        self.console.print(printable)

    def start_initialise_shell_processor(self):
        printable = self.wrap_title(constants.CONSOLE_START_INIT_SHELL_PROCCESS)
        self.console.print(printable)

    def print_initialized_global_processor(self, pandda_args: PanDDAArgs):
        printable = self.indent_text(f"Started the shell processor in mode {pandda_args.global_processing}")
        self.console.print(printable)

        if pandda_args.global_processing == "distributed":
            printable = self.indent_text(f"Distributed options are:")
            self.console.print(printable)

            printable = self.indent_text(
                (
                    f"Tmp directory: {pandda_args.distributed_tmp} \n"
                    f"Number of workers: {pandda_args.distributed_num_workers} \n"
                    f"Project (for SGE): {pandda_args.distributed_project} \n"
                    f"Queue (for SGE): {pandda_args.distributed_queue} \n"
                    f"Cores per worker: {pandda_args.distributed_cores_per_worker} \n"
                    f"Memory per core: {pandda_args.distributed_mem_per_core} \n"
                ),
                indent=8)
            self.console.print(printable)

    def start_initialise_multiprocessor(self):
        printable = self.wrap_title(constants.CONSOLE_START_INIT_MULTIPROCESS)
        self.console.print(printable)

    def print_initialized_local_processor(self, pandda_args: PanDDAArgs):
        printable = self.indent_text(f"Started the multiprocessor in mode {pandda_args.local_processing}")
        self.console.print(printable)

        if pandda_args.local_processing != "serial":
            printable = self.indent_text(f"Multiprocessing options are:")
            self.console.print(printable)

            printable = self.indent_text(
                (
                    f"Number of CPUs: {pandda_args.local_cpus} \n"
                ),
                indent=8)
            self.console.print(printable)

    def start_fs_model(self):
        printable = self.wrap_title(constants.CONSOLE_START_FS_MODEL)
        self.console.print(printable)

    @staticmethod
    def indent_text(text, indent=4):
        return Padding(
            text,
            (0, 0, 0, indent)
        )

    def summarise_fs_model(self, pandda_fs_model: PanDDAFSInterface):
        printable = self.indent_text(
            f"Number of datasets found: {len(pandda_fs_model.input.dataset_dirs)}",
        )
        self.console.print(printable)

    def start_load_datasets(self):
        printable = self.wrap_title(constants.CONSOLE_START_LOAD_DATASETS)
        self.console.print(printable)

    def start_data_quality_filters(self):
        printable = self.wrap_title(constants.CONSOLE_START_QUALITY_FILTERS)
        self.console.print(printable)

    # def start_reference_selection(self):
    #     printable = self.wrap_title(constants.CONSOLE_START_REF_SELEC)
    #     self.console.print(printable)
    #
    # def summarise_reference(self, reference):
    #     printable = self.indent_text(f"Reference dataset is: {str(reference.dtag)}")
    #     self.console.print(printable)
    #
    #     st = reference.dataset.structure.structure.clone()
    #     structure_poss = []
    #     for model in st:
    #         for chain in model:
    #             for residue in chain:
    #                 for atom in residue:
    #                     pos = atom.pos
    #                     structure_poss.append([pos.x, pos.y, pos.z])
    #
    #     pos_array = np.array(structure_poss)
    #     min_pos = np.min(pos_array, axis=0)
    #     max_pos = np.max(pos_array, axis=0)
    #
    #     printable = self.indent_text(f"Reference model min pos: {min_pos[0]} {min_pos[1]} {min_pos[2]}")
    #     self.console.print(printable)
    #
    #     printable = self.indent_text(f"Reference model max pos: {max_pos[0]} {max_pos[1]} {max_pos[2]}")
    #     self.console.print(printable)

    # def start_b_factor_smoothing(self):
    #     printable = self.wrap_title(constants.CONSOLE_START_B_FACTOR_SMOOTHING)
    #     self.console.print(printable)
    #
    # def summarise_b_factor_smoothing(self, datasets: Dict[str, DatasetInterface]):
    #     event_class_table = Table(show_header=True, header_style="bold magenta", expand=True)
    #     event_class_table.title = "Smoothing Factors"
    #     event_class_table.add_column("Dtag")
    #     event_class_table.add_column("Smoothing Factor")
    #
    #     for dtag, dataset in datasets.items():
    #         event_class_table.add_row(
    #             str(dtag.dtag),
    #             str(round(dataset.smoothing_factor, 2)),
    #         )
    #
    #     self.console.print(event_class_table)

    def start_reference_comparability_filters(self):
        printable = self.wrap_title(constants.CONSOLE_START_REF_COMPAT_FILTERS)
        self.console.print(printable)

    def start_get_grid(self):
        printable = self.wrap_title(constants.CONSOLE_START_GET_GRID)
        self.console.print(printable)

    def summarise_get_grid(self, grid):
        printable = self.indent_text(f"Got Grid. Paramaters are:")
        self.console.print(printable)

        printable = self.indent_text(
            f"nu: {grid.spacing[0]} nv: {grid.spacing[1]} nw: {grid.spacing[2]}",
            indent=8,
        )
        self.console.print(printable)

        printable = self.indent_text(
            f"a: {grid.unit_cell[0]} b: {grid.unit_cell[1]} c: {grid.unit_cell[2]}",
            indent=8,
        )
        self.console.print(printable)

        printable = self.indent_text(
            f"alpha: {grid.unit_cell[3]} beta: {grid.unit_cell[4]} gamma: {grid.unit_cell[5]}",
            indent=8,
        )
        self.console.print(printable)

    def start_alignments(self):
        printable = self.wrap_title(constants.CONSOLE_START_ALIGN)
        self.console.print(printable)

    def summarise_local_alignment(self, ):
        printable = self.indent_text(f"Locally aligned all datasets to reference")
        self.console.print(printable)

    def start_get_comparators(self):
        printable = self.wrap_title(constants.CONSOLE_START_GET_COMPAR)
        self.console.print(printable)

    def summarise_get_comarators(self, comparators):
        # self.console.print(comparators)
        first_set = list(comparators.values())[0]

        printable = self.indent_text(f"Found {len(first_set)} statistical models.")
        self.console.print(printable)

    def start_get_shells(self):
        printable = self.wrap_title(f"Getting shells...")
        self.console.print(printable)

    def summarise_get_shells(self, shells):
        printable = self.indent_text(f"Got {len(shells)} for processing. Shells are:")
        self.console.print(printable)

        for shell_number, shell in shells.items():
            printable = self.indent_text(
                (
                    f"{round(shell.res, 2)}: "
                    f"Num test dtags: {len(shell.test_dtags)} "
                    f"Total num dtags: {len(shell.all_dtags)}"
                ),
                indent=8)
            self.console.print(printable)

    def start_process_shells(self):
        printable = self.wrap_title(constants.CONSOLE_START_PROCESS_SHELLS)
        self.console.print(printable)

    def print_starting_process_shell(self, shell):
        printable = self.wrap_title("Processing Shell!")
        self.console.print(printable)

        printable = self.indent_text(f"Processing shell at resolution: {shell.res}")
        self.console.print(printable)

        printable = self.indent_text(f"There are {len(shell.test_dtags)} datasets to analyse in this shell. These are:")
        self.console.print(printable)

        for dtag in shell.test_dtags:
            printable = self.indent_text(f"{dtag.dtag}", indent=8)
            self.console.print(printable)

    def print_starting_truncating_shells(self):
        printable = self.wrap_title(f"Truncating shell datasets")
        self.console.print(printable)

    def print_summarise_truncating_shells(self, shell_truncated_datasets: Dict[str, DatasetInterface]):
        printable = self.indent_text(f"Truncated {len(shell_truncated_datasets)} datasets for processing...")
        self.console.print(printable)

    def print_starting_loading_xmaps(self):
        printable = self.wrap_title(f"Loading xmaps")
        self.console.print(printable)

    def print_summarise_loading_xmaps(self, xmaps, xmap_processing_time: float):
        printable = self.indent_text(f"Loaded {len(xmaps)} aligned XMaps in {xmap_processing_time}")
        self.console.print(printable)

    def print_starting_get_models(self):
        printable = self.wrap_title(f"Getting models")
        self.console.print(printable)

    def print_summarise_get_models(self, models, get_models_time: float):
        printable = self.indent_text(f"Got {len(models)} models in {get_models_time} seconds!")
        self.console.print(printable)

    def print_starting_process_datasets(self):
        printable = self.wrap_title(f"Processing test datasets")
        self.console.print(printable)

    def print_summarise_process_datasets(self, shell_result):
        print(f"\tProcessed test datasets!")

    def start_autobuilding(self):
        printable = self.wrap_title(constants.CONSOLE_START_AUTOBUILDING)
        self.console.print(printable)

    def summarise_autobuilding(self, autobuild_results):
        printable = self.indent_text(f"Autobuilt all event maps")
        self.console.print(printable)

    def summarise_autobuild_model_update(self, dataset_selected_events):
        event_class_table = Table(show_header=True, header_style="bold magenta", expand=True)
        event_class_table.title = "Selected Autobuilds"
        event_class_table.add_column("Dtag")
        event_class_table.add_column("Event ID")
        event_class_table.add_column("Autobuild Path")

        for _dtag in dataset_selected_events:
            dtag, event_idx, path = dataset_selected_events[_dtag]
            event_class_table.add_row(
                str(dtag),
                str(int(event_idx)),
                Path(path).name
            )

        self.console.print(event_class_table)

    def start_autobuild_model_update(self, ):
        printable = self.wrap_title("Updating PanDDA Models With Best Autobuilds")
        self.console.print(printable)

    def start_ranking(self):
        printable = self.wrap_title(constants.CONSOLE_START_RANKING)
        self.console.print(printable)

    def start_assign_sites(self):
        printable = self.wrap_title(constants.CONSOLE_START_ASSIGN_SITES)
        self.console.print(printable)

    def start_run_summary(self):
        printable = self.wrap_title(constants.CONSOLE_START_SUMMARY)
        self.console.print(printable)

    def start_event_table_output(self):
        printable = self.wrap_title(f"Writing Event Table")
        self.console.print(printable)

    def summarise_event_table_output(self, path):
        printable = self.indent_text(f"Event table written to: {str(path)}")
        self.console.print(printable)

    def start_site_table_output(self):
        printable = self.wrap_title(f"Writing Site Table")
        self.console.print(printable)

    def summarise_site_table_output(self, path):
        printable = self.indent_text(f"Site table written to: {str(path)}")
        self.console.print(printable)

    def start_log_save(self):
        printable = self.wrap_title(f"Saving JSON log of run...")
        self.console.print(printable)

    def summarise_log_save(self, path):
        printable = self.indent_text(f"JSON log written to: {str(path)}")
        self.console.print(printable)

    def summarise_run(self, time):
        printable = self.wrap_title(f"PanDDA ran in {time}")
        self.console.print(printable)

    def start_classification(self):
        printable = self.wrap_title(constants.CONSOLE_START_EVENT_CLASSIFICATION)
        self.console.print(printable)

    def summarise_autobuilds(self, autobuild_results):
        event_class_table = Table(show_header=True, header_style="bold magenta", expand=True)
        event_class_table.title = "Autobuild Scores"
        event_class_table.add_column("Dtag")
        event_class_table.add_column("Event Number")
        event_class_table.add_column("Autobuild Score")

        for event_id, autobuild_result in autobuild_results.items():
            selected_build = autobuild_result.selected_fragment_path

            if not selected_build:
                build_score_string = "None"
            else:

                selected_build_score = autobuild_result.scores[selected_build]
                build_score_string = str(round(selected_build_score, 2))

            event_class_table.add_row(
                str(event_id.dtag.dtag),
                str(event_id.event_idx.event_idx),
                build_score_string,
            )

        self.console.print(event_class_table)

    def start_rescoring(self, rescore_event_method: str):
        printable = self.wrap_title("Rescoring Events...")
        self.console.print(printable)

        printable = self.indent_text(f"Rescoring method is: {str(rescore_event_method)}")
        self.console.print(printable)

    # def summarise_rescoring(self, event_scores: EventScoresInterface):
    #     printable = self.indent_text(f"Rescored events. Printing new event score table.")
    #     self.console.print(printable)
    #
    #     event_table = Table(show_header=True, header_style="bold magenta", expand=True)
    #     event_table.title = "Event Scores"
    #     event_table.add_column("Dtag")
    #     event_table.add_column("Event Number")
    #     event_table.add_column("Event Score")
    #
    #     for event_id, event_score in event_scores.items():
    #         event_table.add_row(
    #             str(event_id.dtag.dtag),
    #             str(event_id.event_idx.event_idx),
    #             str(round(float(event_score), 2)),
    #         )
    #
    #     self.console.print(event_table)

    # def summarise_event_classifications(self, event_classifications: EventClassificationsInterface):
    #     event_class_table = Table(show_header=True, header_style="bold magenta", expand=True)
    #     event_class_table.title = "Event Classifications"
    #     event_class_table.add_column("Dtag")
    #     event_class_table.add_column("Event Number")
    #     event_class_table.add_column("Class")
    #
    #     for event_id, event_class in event_classifications.items():
    #         event_class_table.add_row(
    #             str(event_id.dtag.dtag),
    #             str(event_id.event_idx.event_idx),
    #             str(event_class),
    #         )
    #
    #     self.console.print(event_class_table)

    def summarise_datasets(self, datasets_initial: Dict[str, DatasetInterface], fs: PanDDAFSInterface):
        # Statistics
        # printable = self.indent_text(
        #     f"Unit cell statistics"
        # )
        # self.console.print(printable)

        # Unit cells
        unit_cell_table = Table(show_header=True, header_style="bold magenta", expand=True)
        unit_cell_table.title = "Unit Cell Statistics"
        unit_cell_table.add_column("Parameter")
        unit_cell_table.add_column("Min")
        unit_cell_table.add_column("Mean")
        unit_cell_table.add_column("Max")

        unit_cells = [dataset.structure.structure.cell for dataset in datasets_initial.values()]

        uca = [uc.a for uc in unit_cells]
        ucb = [uc.b for uc in unit_cells]
        ucc = [uc.c for uc in unit_cells]
        ucalpha = [uc.alpha for uc in unit_cells]
        ucbeta = [uc.beta for uc in unit_cells]
        ucgamma = [uc.gamma for uc in unit_cells]
        unit_cell_table.add_row("a", str(round(np.min(uca), 2)), str(round(np.mean(uca), 2)),
                                str(round(np.max(uca), 2)))
        unit_cell_table.add_row("b", str(round(np.min(ucb), 2)), str(round(np.mean(ucb), 2)),
                                str(round(np.max(ucb), 2)))
        unit_cell_table.add_row("c", str(round(np.min(ucc), 2)), str(round(np.mean(ucc), 2)),
                                str(round(np.max(ucc), 2)))
        unit_cell_table.add_row("alpha", str(round(np.min(ucalpha), 2)), str(round(np.mean(ucalpha), 2)),
                                str(round(np.max(ucalpha), 2)))
        unit_cell_table.add_row("beta", str(round(np.min(ucbeta), 2)), str(round(np.mean(ucbeta), 2)),
                                str(round(np.max(ucbeta), 2)))
        unit_cell_table.add_row("gamma", str(round(np.min(ucgamma), 2)), str(round(np.mean(ucgamma), 2)),
                                str(round(np.max(ucgamma), 2)))

        self.console.print(unit_cell_table)

        # Resolutions
        ress = [dataset.reflections.resolution() for dataset in datasets_initial.values()]

        resolution_table = Table(show_header=True, header_style="bold magenta", expand=True)
        resolution_table.title = "Resolutions"
        resolution_table.add_column("Min")
        resolution_table.add_column("Mean")
        resolution_table.add_column("Max")
        resolution_table.add_row(str(round(np.min(ress), 2)), str(round(np.mean(ress), 2)), str(round(np.max(ress), 2)))

        self.console.print(
            resolution_table
        )

        # Spacegroups
        sgtable = Table(show_header=True, header_style="bold magenta", expand=True)
        sgtable.title = "Spacegroups"
        sgtable.add_column("Spacegroup")
        sgtable.add_column("Count")

        sgs = [dataset.structure.structure.spacegroup_hm for dataset in datasets_initial.values()]

        values, counts = np.unique(sgs, return_counts=True)
        for sg, count in zip(values, counts):
            sgtable.add_row(sg, str(count))

        self.console.print(sgtable)

        # Chains
        chain_table = Table(show_header=True, header_style="bold magenta", expand=True)
        chain_table.title = "Chains"
        chain_table.add_column("Chains")
        chain_table.add_column("Count")

        chains = []
        for dataset in datasets_initial.values():
            _chains = []
            for model in dataset.structure.structure:
                for chain in model:
                    _chains.append(chain.name)

        values, counts = np.unique([" ".join(chains) for chains in chains], return_counts=True)
        for chains, count in zip(values, counts):
            chain_table.add_row(chains, str(count))
        self.console.print(chain_table)

        # Datasets
        table = Table(show_header=True, header_style="bold magenta", expand=True)
        table.title = "Datasets"
        # Columns
        table.add_column("Dtag")
        table.add_column("Resolution")
        table.add_column("Spacegroup")
        table.add_column("SMILES?")
        table.add_column("CIF?")
        table.add_column("PDB?")

        # Rows
        for dtag in sorted(datasets_initial, key=lambda x: x):
            dataset = datasets_initial[dtag]
            has_smiles = False
            for ligand_key, ligand_files in dataset.ligand_files.items():
                if ligand_files.ligand_smiles:
                    has_smiles = True
            # if fs.processed_datasets.processed_datasets[dtag].input_ligand_smiles.exists():
            #     has_smiles = True

            has_cif = False
            for ligand_key, ligand_files in dataset.ligand_files.items():
                if ligand_files.ligand_cif:
                    has_cif = True
            # if fs.processed_datasets.processed_datasets[dtag].input_ligand_cif.exists():
            #     has_cif = True

            has_pdb = False
            for ligand_key, ligand_files in dataset.ligand_files.items():
                if ligand_files.ligand_pdb:
                    has_pdb = True
            # if fs.processed_datasets.processed_datasets[dtag].input_ligand_pdb.exists():
            #     has_pdb = True

            table.add_row(
                dtag,
                str(round(dataset.reflections.reflections.resolution_high(), 2)),
                dataset.reflections.reflections.spacegroup.hm,
                str(has_smiles),
                str(has_cif),
                str(has_pdb)
            )

        self.console.print(table)

    def summarize_datasets_to_process(self, dataset_to_process, datasets_not_to_process):
        self.wrap_subtitle(f"Datasets to Process")
        self.console.print(self.indent_text(Columns(sorted(dataset_to_process)), indent=4))

        self.wrap_subtitle(f"Datasets not to Process")
        for dtag in sorted(datasets_not_to_process):
            self.console.print(self.indent_text(f"{dtag} : {datasets_not_to_process[dtag]}"))

    def begin_dataset_processing(
            self,
            dtag: str,
            dataset: DatasetInterface,
            dataset_res: float,
            comparator_datasets: Dict[str, DatasetInterface],
            processing_res: float,
            j: int,
            dataset_to_process: List,
            time_begin_process_datasets: float
    ):
        printable = self.wrap_title(f"{dtag} : {j+1} / {len(dataset_to_process)}")
        self.console.print(printable)

        printable = self.indent_text(f"Resolution: {round(dataset_res, 2)}")
        self.console.print(printable)

        printable = self.indent_text(f"Processing Resolution: {round(processing_res, 2)}")
        self.console.print(printable)

        self.wrap_subtitle(f"Comparator Datasets")
        printable = self.indent_text(Columns([x for x in sorted(comparator_datasets)]))
        self.console.print(printable)

        estimated_time_per_dataset = (time.time() - time_begin_process_datasets) / (j+1)
        estimated_time_to_completion = (len(dataset_to_process) - (j+1)) * estimated_time_per_dataset
        self.wrap_subtitle(f"Estimated time to completion: {round(estimated_time_to_completion, 2)} seconds!")


    def insufficient_comparators(self, comparator_datasets):
        printable = self.indent_text(f"NOT ENOUGH COMPARATOR DATASETS: {len(comparator_datasets)}! SKIPPING!")
        self.console.print(printable)

    def no_ligand_data(self):
        printable = self.indent_text(f"No ligand files for this dataset! Skipping!")
        self.console.print(printable)

    def processed_dataset(
            self,
            dtag: str,
            dataset: DatasetInterface,
            comparator_datasets: Dict[str, DatasetInterface],
            processing_res: float,
            characterization_sets: Dict[int, Dict[str, DatasetInterface]],
            models_to_process: List[int],
            processed_models: Dict[int, Tuple[Dict[Tuple[str, int], EventInterface], Any, Any]],
            selected_model_num: int,
            selected_model_events: Dict[int, EventInterface],
    ):

        self.wrap_subtitle("Ligand Files")
        for ligand_key, ligand_files in dataset.ligand_files.items():
            self.console.print(self.indent_text(f"Ligand Key: {ligand_key}"))
            self.console.print(self.indent_text(f"{ligand_files.ligand_pdb}", indent=8))
            self.console.print(self.indent_text(f"{ligand_files.ligand_cif}", indent=8))
            self.console.print(self.indent_text(f"{ligand_files.ligand_smiles}", indent=8))

        self.wrap_subtitle("Model Information")
        self.console.print(self.indent_text(f"Processed Models: {models_to_process}"))
        self.console.print(self.indent_text(f"Selected model: {selected_model_num}"))
        # self.console.print(self.indent_text(f"Got {len(comparator_datasets)} comparator datasets"))

        # self.console.print(self.indent_text(f"Dataset processed at resolution: {processing_res}"))

        # self.console.print(self.indent_text(f"Model Information"))
        for model_number, characterization_set in characterization_sets.items():
            self.console.print(self.indent_text(f"Model Number: {model_number}"))
            if model_number in models_to_process:
                self.console.print(self.indent_text(f"Processed: True", indent=8))
            else:
                self.console.print(self.indent_text(f"Processed: False", indent=8))
            self.console.print(self.indent_text(Columns(sorted(characterization_set)), indent=8))

        self.wrap_subtitle(f"Processed Model Results")
        # self.console.print(self.indent_text(f"Processed Model Results:"))
        for model_number, processed_model in processed_models.items():
            events, z, mean, std, meta = processed_models[model_number]
            self.console.print(self.indent_text(f"Model Number: {model_number}"))
            if events:
                self.console.print(self.indent_text(f"Number of events: {len(events)}", indent=8))
            else:
                self.console.print(self.indent_text(f"Number of events: 0", indent=8))

    def processed_autobuilds(self, autobuilds: Dict[Tuple[str, int], Dict[str, AutobuildInterface]]):
        nested_autobuilds = {}
        for event_id, ligand_autobuild_results in autobuilds.items():
            dtag, event_idx = event_id
            if dtag not in nested_autobuilds:
                nested_autobuilds[dtag] = {}

            nested_autobuilds[dtag][event_idx] = {}

            for ligand_key, autobuild_results in ligand_autobuild_results.items():
                nested_autobuilds[dtag][event_idx][ligand_key] = autobuild_results

        for dtag in sorted(nested_autobuilds):
            self.wrap_subtitle(dtag)
            dtag_events = nested_autobuilds[dtag]
            for event_idx in sorted(dtag_events):
                event_ligand_autobuilds: Dict[str, AutobuildInterface] = dtag_events[event_idx]
                for ligand_key, autobuild_results in event_ligand_autobuilds.items():
                    if autobuild_results:
                        if len(autobuild_results.log_result_dict) > 0:
                            max_score = max(autobuild_results.log_result_dict,
                                            key=lambda x: autobuild_results.log_result_dict[x])
                            printable = self.indent_text(f"{max_score}: {autobuild_results.log_result_dict[max_score]}")
                            self.console.print(printable)

    def start_identify_structure_factors(self, ):
        printable = self.wrap_title("Getting Structure Factors...")
        self.console.print(printable)

    def summarise_structure_factors(self, structure_factors, label_counts):
        printable = self.indent_text(f"Label counts are: ")
        self.console.print(printable)

        printable = self.indent_text(Pretty(label_counts), indent=8)
        self.console.print(printable)

        printable = self.indent_text(f"Structure factors are: {structure_factors.f} {structure_factors.phi}")
        self.console.print(printable)

    def summarise_shells(self,
                         shell_results,
                         events,
                         event_scores,
                         ):
        event_table = Table(show_header=True, header_style="bold magenta", expand=True)
        event_table.title = "Shell Events"
        event_table.add_column("Res")
        event_table.add_column("Dtag")
        event_table.add_column("Event Number")
        event_table.add_column("Event Score")
        event_table.add_column("Event Size")

        for res, shell_result in shell_results.items():
            res = shell_result.shell.res
            dataset_results = shell_result.dataset_results
            for dtag, dataset_result in dataset_results.items():
                dataset_events = dataset_result.events
                dataset_event_scores = dataset_result.event_scores
                for event_id, event in dataset_events.items():
                    event_score = dataset_event_scores[event_id]
                    selected_structure_score = event_score.get_selected_structure_score()
                    if selected_structure_score:
                        score = round(selected_structure_score, 2)
                    else:
                        score = None
                    event_table.add_row(
                        str(round(res, 2)),
                        str(event_id.dtag.dtag),
                        str(event_id.event_idx.event_idx),
                        str(score),
                        str(event.cluster.indexes[0].size)
                    )

        self.console.print(event_table)

    def summarise_sites(self, sites):
        event_class_table = Table(show_header=True, header_style="bold magenta", expand=True)
        event_class_table.title = "Event Classifications"
        event_class_table.add_column("Dtag")
        event_class_table.add_column("Event Number")
        event_class_table.add_column("Site")

        for event_id, site_id in sites.event_to_site.items():
            event_class_table.add_row(
                str(event_id.dtag.dtag),
                str(event_id.event_idx.event_idx),
                str(site_id.site_id),
            )

        self.console.print(event_class_table)

    def summarise_filtered_datasets(self, filtered_dtags: Dict[str, List[str]]):
        for filter_key, filtered in filtered_dtags.items():
            self.console.print(f"Filtered with {filter_key}: {filtered}")

    def print_exception(self, ):
        self.console.print_exception()

    def save(self, console_log_file):
        self.console.save_text(str(console_log_file))

    def __enter__(self):
        ...

    def __exit__(self, exc_type, exc_val, exc_tb):
        ...


def get_pandda_console():
    return PanDDAConsole()


