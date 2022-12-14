import numpy as np

from rich.console import Console
from rich.panel import Panel
from rich.align import Align
from rich.padding import Padding
from rich.table import Table
from rich.pretty import Pretty

from pandda_gemmi.analyse_interface import *
from pandda_gemmi import constants
from pandda_gemmi.args import PanDDAArgs

import subprocess


def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


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
        printable = self.indent_text(f"Started the shell processor in mode {pandda_args.local_processing}")
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

    def summarise_fs_model(self, pandda_fs_model):
        printable = self.indent_text(
            f"Number of datasets found: {len(pandda_fs_model.processed_datasets.processed_datasets)}",
        )
        self.console.print(printable)

    def start_load_datasets(self):
        printable = self.wrap_title(constants.CONSOLE_START_LOAD_DATASETS)
        self.console.print(printable)

    def start_data_quality_filters(self):
        printable = self.wrap_title(constants.CONSOLE_START_QUALITY_FILTERS)
        self.console.print(printable)

    def start_reference_selection(self):
        printable = self.wrap_title(constants.CONSOLE_START_REF_SELEC)
        self.console.print(printable)

    def summarise_reference(self, reference):
        self.console.print(str(reference.dtag))

    def start_b_factor_smoothing(self):
        printable = self.wrap_title(constants.CONSOLE_START_B_FACTOR_SMOOTHING)
        self.console.print(printable)

    def summarise_b_factor_smoothing(self, datasets: DatasetsInterface):
        event_class_table = Table(show_header=True, header_style="bold magenta", expand=True)
        event_class_table.title = "Smoothing Factors"
        event_class_table.add_column("Dtag")
        event_class_table.add_column("Smoothing Factor")

        for dtag, dataset in datasets.items():
            event_class_table.add_row(
                str(dtag.dtag),
                round(dataset.smoothing_factor, 2),
            )

        self.console.print(event_class_table)

    def start_reference_comparability_filters(self):
        printable = self.wrap_title(constants.CONSOLE_START_REF_COMPAT_FILTERS)
        self.console.print(printable)

    def start_get_grid(self):
        printable = self.wrap_title(constants.CONSOLE_START_GET_GRID)
        self.console.print(printable)

    def summarise_get_grid(self, grid: GridInterface):
        printable = self.indent_text(f"Got Grid. Paramaters are:")
        self.console.print(printable)

        printable = self.indent_text(f"nu: {grid.grid.nu} nv: {grid.grid.nv} nw: {grid.grid.nw}", indent=8)
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
                    f"{shell.res}: "
                    f"Num test dtags: {len(shell.test_dtags)} "
                    f"Total num dtags: {len(shell.all_dtags)}"
                ),
                indent=8)
            self.console.print(printable)


def start_process_shells(self):
    printable = self.wrap_title(constants.CONSOLE_START_PROCESS_SHELLS)
    self.console.print(printable)


def start_autobuilding(self):
    printable = self.wrap_title(constants.CONSOLE_START_AUTOBUILDING)
    self.console.print(printable)


def summarise_autobuilding(self, autobuild_results: AutobuildResultsInterface):
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
            int(event_idx),
            path.name
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


def summarise_autobuilds(self, autobuild_results: AutobuildResultsInterface):
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


def summarise_event_classifications(self, event_classifications: EventClassificationsInterface):
    event_class_table = Table(show_header=True, header_style="bold magenta", expand=True)
    event_class_table.title = "Event Classifications"
    event_class_table.add_column("Dtag")
    event_class_table.add_column("Event Number")
    event_class_table.add_column("Class")

    for event_id, event_class in event_classifications.items():
        event_class_table.add_row(
            str(event_id.dtag.dtag),
            str(event_id.event_idx.event_idx),
            str(event_class),
        )

    self.console.print(event_class_table)


def summarise_datasets(self, datasets_initial, dataset_statistics):
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
    unit_cell_table.add_column("Max")
    unit_cell_table.add_column("Mean")

    uca = dataset_statistics.unit_cells["a"]
    ucb = dataset_statistics.unit_cells["b"]
    ucc = dataset_statistics.unit_cells["c"]
    ucalpha = dataset_statistics.unit_cells["alpha"]
    ucbeta = dataset_statistics.unit_cells["beta"]
    ucgamma = dataset_statistics.unit_cells["gamma"]
    unit_cell_table.add_row("a", str(np.min(uca)), str(np.mean(uca)), str(np.max(uca)))
    unit_cell_table.add_row("b", str(np.min(ucb)), str(np.mean(ucb)), str(np.max(ucb)))
    unit_cell_table.add_row("c", str(np.min(ucc)), str(np.mean(ucc)), str(np.max(ucc)))
    unit_cell_table.add_row("alpha", str(np.min(ucalpha)), str(np.mean(ucalpha)), str(np.max(ucalpha)))
    unit_cell_table.add_row("beta", str(np.min(ucbeta)), str(np.mean(ucbeta)), str(np.max(ucbeta)))
    unit_cell_table.add_row("gamma", str(np.min(ucgamma)), str(np.mean(ucgamma)), str(np.max(ucgamma)))

    self.console.print(unit_cell_table)

    # Resolutions
    ress = dataset_statistics.resolutions

    resolution_table = Table(show_header=True, header_style="bold magenta", expand=True)
    resolution_table.title = "Resolutions"
    resolution_table.add_column("Min")
    resolution_table.add_column("Mean")
    resolution_table.add_column("Max")
    resolution_table.add_row(str(np.min(ress)), str(np.mean(ress)), str(np.max(ress)))

    self.console.print(
        resolution_table
    )

    # Spacegroups
    sgtable = Table(show_header=True, header_style="bold magenta", expand=True)
    sgtable.title = "Spacegroups"
    sgtable.add_column("Spacegroup")
    sgtable.add_column("Count")
    values, counts = np.unique(dataset_statistics.spacegroups, return_counts=True)
    for sg, count in zip(values, counts):
        sgtable.add_row(sg, str(count))

    self.console.print(sgtable)

    # Chains
    chain_table = Table(show_header=True, header_style="bold magenta", expand=True)
    chain_table.title = "Chains"
    chain_table.add_column("Chains")
    chain_table.add_column("Count")
    values, counts = np.unique([" ".join(chains) for chains in dataset_statistics.chains], return_counts=True)
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

    # Rows
    for dtag in sorted(datasets_initial, key=lambda x: x.dtag):
        dataset = datasets_initial[dtag]
        table.add_row(
            dtag.dtag,
            str(dataset.reflections.reflections.resolution_high()),
            dataset.reflections.reflections.spacegroup.hm,
        )

    self.console.print(table)


def start_identify_structure_factors(self, ):
    printable = self.wrap_title("Getting Structure Factors...")
    self.console.print(printable)


def summarise_structure_factors(self, structure_factors, label_counts):
    printable = self.indent_text(f"label counts are: ")
    self.console.print(printable)

    printable = self.indent_text(Pretty(label_counts), indent=8)
    self.console.print(printable)

    printable = self.indent_text(f"Structure factors are: {structure_factors.f} {structure_factors.phi}")
    self.console.print(printable)


def summarise_shells(self,
                     shell_results: ShellResultsInterface,
                     events: EventsInterface,
                     event_scores: EventScoresInterface,
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


def summarise_filtered_datasets(self, filtered_dtags: Dict[str, List[DtagInterface]]):
    for filter_key, filtered in filtered_dtags.items():
        self.console.print(f"Filtered with {filter_key}: {filtered}")


def print_exception(self, ):
    self.console.print_exception()


def save(self, console_log_file):
    self.console.save_html(str(console_log_file))


def __enter__(self):
    ...


def __exit__(self, exc_type, exc_val, exc_tb):
    ...


def get_pandda_console():
    return PanDDAConsole()
