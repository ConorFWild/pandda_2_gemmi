import numpy as np

from rich.console import Console
from rich.panel import Panel
from rich.align import Align
from rich.padding import Padding
from rich.table import Table

from pandda_gemmi.analyse_interface import *
from pandda_gemmi import constants


class PanDDAConsole:

    def __init__(self):
        self.console = Console(record=True)

    def summarise_arguments(self, args):
        self.console.print(args)

    def wrap_title(self, string):
        return Panel(Align.center(f"[bold]{string}"))

    def start_dependancy_check(self):
        printable = self.wrap_title(constants.CONSOLE_START_DEPENDENCY_CHECK)
        self.console.print(printable)

    def start_log(self):
        printable = self.wrap_title(constants.CONSOLE_START_LOG)
        self.console.print(printable)

    def start_initialise_shell_processor(self):
        printable = self.wrap_title(constants.CONSOLE_START_INIT_SHELL_PROCCESS)
        self.console.print(printable)

    def start_initialise_multiprocessor(self):
        printable = self.wrap_title(constants.CONSOLE_START_INIT_MULTIPROCESS)
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

    def start_reference_comparability_filters(self):
        printable = self.wrap_title(constants.CONSOLE_START_REF_COMPAT_FILTERS)
        self.console.print(printable)

    def start_get_grid(self):
        printable = self.wrap_title(constants.CONSOLE_START_GET_GRID)
        self.console.print(printable)

    def start_alignments(self):
        printable = self.wrap_title(constants.CONSOLE_START_ALIGN)
        self.console.print(printable)

    def start_get_comparators(self):
        printable = self.wrap_title(constants.CONSOLE_START_GET_COMPAR)
        self.console.print(printable)

    def start_process_shells(self):
        printable = self.wrap_title(constants.CONSOLE_START_PROCESS_SHELLS)
        self.console.print(printable)

    def start_autobuilding(self):
        printable = self.wrap_title(constants.CONSOLE_START_AUTOBUILDING)
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


    def print_exception(self,):
        self.console.print_exception()

    def save(self, console_log_file):
        self.console.save_html(str(console_log_file))

    def __enter__(self):
        ...

    def __exit__(self, exc_type, exc_val, exc_tb):
        ...


def get_pandda_console():
    return PanDDAConsole()
