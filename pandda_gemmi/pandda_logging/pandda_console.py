import numpy as np

from rich.console import Console
from rich.panel import Panel
from rich.align import Align
from rich.padding import Padding
from rich.table import Table


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
            (0,0, 0, indent)
        )

    def summarise_fs_model(self, pandda_fs_model):
        printable = self.indent_text(
            f"Number of datasets found: {len(pandda_fs_model.processed_datasets.processed_datasets)}",
        )
        self.console.print(printable)

    def start_load_datasets(self):
        printable = self.wrap_title(constants.CONSOLE_START_LOAD_DATASETS)
        self.console.print(printable)

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
        self.console.print(
            Panel(
                f"Resolution range is: min {np.min(ress)} : mean {np.mean(ress)} : max {np.max(ress)}",
                title="[bold]Resolutions"
            )
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
        for dtag in sorted(datasets_initial, key = lambda x: x.dtag):
            dataset = datasets_initial[dtag]
            table.add_row(
                dtag.dtag,
                str(dataset.reflections.reflections.resolution_high()),
                dataset.reflections.reflections.spacegroup.hm,
            )


        self.console.print(table)

    def print_exception(self, e, debug):
        if debug:
            self.console.print_exception()
        else:
            self.console.print(e)

    def save(self, console_log_file):
        self.console.save_html(str(console_log_file))

    def __enter__(self):
        ...

    def __exit__(self, exc_type, exc_val, exc_tb):
        ...