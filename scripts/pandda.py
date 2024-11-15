from pandda_gemmi.args import PanDDAArgs
from pandda_gemmi.pandda.pandda import pandda

if __name__ == '__main__':
    # Parse Command Line Arguments
    args = PanDDAArgs.from_command_line()

    # Process the PanDDA
    pandda(args)
