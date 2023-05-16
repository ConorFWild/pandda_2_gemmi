from pathlib import Path

from .pandda_input import PanDDAInput
from .pandda_output import PanDDAOutput
class PanDDAFS:
    def __init__(self, input_dir: Path, output_dir: Path):
        # Parse the input directory to extract datasets of structures, reflections and ligand files
        self.input = PanDDAInput(input_dir)

        # Define the output directory structure (such as it is possible to know at this point)
        self.output = PanDDAOutput(output_dir, self.input)