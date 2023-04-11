from pathlib import Path

from .pandda_input import PanDDAInput
from .pandda_output import PanDDAOutput
class PanDDAFS:
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input = PanDDAInput(input_dir)
        self.output = PanDDAOutput(output_dir, self.input)