from pandda_gemmi.analyse_interface import *


class GetEventClassTrivial(GetEventClassTrivialInterface):

    def __init__(self) -> None:
        self.tag: Literal["trivial"] = "trivial"

    def __call__(self, event: EventInterface, ) -> bool:
        return True
