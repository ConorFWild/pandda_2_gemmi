from pandda_gemmi.analyse_interface import *


class Comparators(MutableMapping[DtagInterface, MutableMapping[int, List[DtagInterface]]]):
    ...

