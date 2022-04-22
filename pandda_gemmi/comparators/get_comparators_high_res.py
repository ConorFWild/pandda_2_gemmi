from pandda_gemmi.analyse_interface import *


def get_comparators_high_res(
        datasets,
        alignments,
        grid,
        structure_factors,
        pandda_fs_model,
        comparison_min_comparators,
        comparison_max_comparators,
):
    dtag_list = [dtag for dtag in datasets]

    dtags_by_res = list(
        sorted(
            dtag_list,
            key=lambda dtag: datasets[dtag].reflections.resolution().resolution,
        )
    )

    highest_res_datasets = dtags_by_res[:comparison_min_comparators + 1]

    comparators = {}
    for dtag in dtag_list:
        comparators[dtag] = highest_res_datasets

    return comparators





class GetComparatorsHighRes(GetComparatorsInterface):
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
    pandda_fs_model: PanDDAFSModelInterface):
        return get_comparators_high_res(
            datasets, 
            alignments, 
            grid, 
            structure_factors, 
            pandda_fs_model,
            self.comparison_min_comparators,
            self.comparison_max_comparators)