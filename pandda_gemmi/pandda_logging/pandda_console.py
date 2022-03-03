from rich.console import Console


class PanDDAConsole:

    def __init__(self):
        self.console = Console()

    def summarise_arguments(self, args):
        self.console.print(args)
