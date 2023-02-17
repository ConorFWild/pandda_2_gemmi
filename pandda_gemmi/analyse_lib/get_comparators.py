from pandda_gemmi.common import update_log
from pandda_gemmi.analyse_interface import *
from pandda_gemmi import constants


def get_comparators(pandda_args, console, pandda_fs_model: PanDDAFSModelInterface, pandda_log, comparators_func, datasets, structure_factors,
                    alignments,
                    grid):
    console.start_get_comparators()
    # with STDOUTManager('Deciding on the datasets to characterise the groundstate for each dataset to analyse...',
    #                    f'\tDone!'):
    # TODO: Fix typing for comparators func
    comparators: ComparatorsInterface = comparators_func(
        datasets,
        alignments,
        grid,
        structure_factors,
        pandda_fs_model,
    )
    # if pandda_args.comparison_strategy == "cluster" or pandda_args.comparison_strategy == "hybrid":
    #     pandda_log["Cluster Assignments"] = {str(dtag): int(cluster) for dtag, cluster in
    #                                          cluster_assignments.items()}
    #     pandda_log["Neighbourhood core dtags"] = {int(neighbourhood_number): [str(dtag) for dtag in
    #                                                                           neighbourhood.core_dtags]
    #                                               for neighbourhood_number, neighbourhood
    #                                               in comparators.items()
    #                                               }
    # if pandda_args.debug:
    #     print("Comparators are:")
    # printer.pprint(pandda_log["Cluster Assignments"])
    # printer.pprint(pandda_log["Neighbourhood core dtags"])
    # printer.pprint(comparators)
    update_log(pandda_log, pandda_args.out_dir / constants.PANDDA_LOG_FILE)
    console.summarise_get_comarators(comparators)
    pandda_fs_model.comparators_file.save(comparators)
    return comparators
