from pandda_gemmi.dataset import StructureFactors
from pandda_gemmi.analyse_interface import *


def get_structure_factors(pandda_args, console, get_common_structure_factors, datasets_initial):
    console.start_identify_structure_factors()
    # with STDOUTManager('Looking for common structure factors in datasets...', f'\tFound structure factors!'):
    label_counts = None
    if not pandda_args.structure_factors:
        potential_structure_factors, label_counts = get_common_structure_factors(
            datasets_initial)
        # If still no structure factors
        if not potential_structure_factors:
            raise Exception(
                "No common structure factors found in mtzs. Please manually provide the labels with the --structure_factors option.")
        else:
            structure_factors: StructureFactorsInterface = potential_structure_factors
    else:
        structure_factors: StructureFactorsInterface = StructureFactors(pandda_args.structure_factors[0],
                                                                        pandda_args.structure_factors[1],
                                                                        )

    console.summarise_structure_factors(structure_factors, label_counts)

    return structure_factors