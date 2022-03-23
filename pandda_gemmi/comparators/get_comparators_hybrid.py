from pandda_gemmi.analyse_interface import *

from pandda_gemmi.comparators import get_multiple_comparator_sets, get_comparators_high_res_first


class GetComparatorsHybrid(GetComparatorsInterface):
    def __init__(self,
                 comparison_min_comparators: int,
                 comparison_max_comparators: int,
                 sample_rate: float,
                 resolution_cutoff: float,
                 load_xmap_flat_func,
                 process_local: ProcessorInterface,
                 debug: bool,
                 ):
        self.comparison_min_comparators = comparison_min_comparators
        self.comparison_max_comparators = comparison_max_comparators
        self.sample_rate = sample_rate
        self.load_xmap_flat_func = load_xmap_flat_func
        self.resolution_cutoff = resolution_cutoff
        self.process_local = process_local
        self.debug = debug

    def __call__(self,
                 datasets: Dict[DtagInterface, DatasetInterface],
                 alignments: Dict[DtagInterface, AlignmentInterface],
                 grid: GridInterface,
                 structure_factors: StructureFactorsInterface,
                 pandda_fs_model: PanDDAFSModelInterface,
                 ) -> ComparatorsInterface:
        comparators_first: ComparatorsInterface = get_comparators_high_res_first(
            datasets,
            alignments,
            grid,
            structure_factors,
            pandda_fs_model,
            self.comparison_min_comparators,
            self.comparison_max_comparators,
        )[0]

        print(comparators_first)

        comparators_multiple, clusters = get_multiple_comparator_sets(
            datasets,
            alignments,
            grid,
            structure_factors,
            pandda_fs_model,
            comparison_min_comparators=self.comparison_min_comparators,
            sample_rate=self.sample_rate,
            resolution_cutoff=self.resolution_cutoff,
            load_xmap_flat_func=self.load_xmap_flat_func,
            process_local=self.process_local,
            debug=self.debug,
        )

        print(comparators_multiple)

        comparators: ComparatorsInterface = {}

        for dtag, dtag_models in comparators_first.items():
            comparators[dtag] = {}
            for model_number, model_comparators in dtag_models.items():
                comparators[dtag][model_number] = model_comparators

        for dtag, dtag_models in comparators_multiple.items():
            for model_number, model_comparators in dtag_models.items():
                comparators[dtag][model_number+1] = model_comparators


        return comparators, clusters

