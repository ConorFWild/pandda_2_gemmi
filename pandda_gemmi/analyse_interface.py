from typing import *


# Analyse class interfaces
class PanDDAConsoleInterface:
    ...


class ProcessorInterface:
    ...


class PanDDAFSModelInterface:
    ...


class StructureFactorInterface:
    ...


# Analyse Function Interfaces
GetPanDDAConsole = Callable[[], PanDDAConsoleInterface]
# GetProcessLocal
# GetProcessGlobal
#
# Smooth
# LoadXMap
# LoadXMapFlat
# AnalyseModel
# GetComparators
#
#
# get_smooth_func
# get_load_xmap_func
# get_load_xmap_flat_func
# get_analyse_model_func