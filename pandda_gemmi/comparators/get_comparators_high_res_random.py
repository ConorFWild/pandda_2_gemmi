from pandda_gemmi.analyse_interface import *


def get_comparators_high_res_random(
        datasets,
        alignments,
        grid,
        structure_factors,
        pandda_fs_model,
        comparison_min_comparators,
        comparison_max_comparators,
) -> ComparatorsInterface:
    dtag_list = [dtag for dtag in datasets]

    dtags_by_res = list(
        sorted(
            dtag_list,
            key=lambda dtag: datasets[dtag].reflections.resolution().resolution,
        )
    )

    highest_res_datasets = dtags_by_res[:comparison_min_comparators + 1]
    highest_res_datasets_max = max(
        [datasets[dtag].reflections.resolution().resolution for dtag in highest_res_datasets])

    comparators = {}
    for dtag in dtag_list:
        current_res = datasets[dtag].reflections.resolution().resolution

        truncation_res = max(current_res, highest_res_datasets_max)

        truncated_datasets = [dtag for dtag in dtag_list if
                              datasets[dtag].reflections.resolution().resolution < truncation_res]

        comparators[dtag] = list(
            np.random.choice(
                truncated_datasets,
                size=comparison_min_comparators,
                replace=False,
            )
        )

    return comparators

class GetComparatorsHighResRandom(GetComparatorsInterface):
    def __init__(self,
            comparison_min_comparators: int,
        comparison_max_comparators: int,
        ) -> None:
        self.comparison_min_comparators: int = comparison_min_comparators
        self.comparison_max_comparators: int = comparison_max_comparators
        

    def __call__(self, 
    datasets: Dict[DtagInterface, DatasetInterface], 
    alignments: Dict[DtagInterface, AlignmentInterface], 
    grid: GridInterface, 
    structure_factors: StructureFactorsInterface, 
    pandda_fs_model: PanDDAFSModelInterface) -> ComparatorsInterface:
        return get_comparators_high_res_random(
            datasets, 
            alignments, 
            grid, 
            structure_factors, 
            pandda_fs_model,
            self.comparison_min_comparators,
            self.comparison_max_comparators)