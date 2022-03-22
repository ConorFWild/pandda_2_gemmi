from pandda_gemmi.analyse_interface import *


class GetEventClassTrivial(GetEventClassInterface):

    def __call__(self, event: EventInterface, ) -> bool:
        return True
