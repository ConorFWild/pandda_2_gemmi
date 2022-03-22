from pandda_gemmi.analyse_interface import *


class GetComparatorsHybrid(GetComparatorsInterface):
    def __init__(self):
        ...

    def __call__(self,
                 datasets: Dict[DtagInterface, DatasetInterface],
                 alignments: Dict[DtagInterface, AlignmentInterface],
                 grid: GridInterface,
                 structure_factors: StructureFactorsInterface,
                 pandda_fs_model: PanDDAFSModelInterface,
                 ):
        ...
