from rich.console import Console
from rich.panel import Panel
from rich.align import Align

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