from pandda_gemmi.analyse_interface import *


class GetEventClassAutobuildScore(GetEventClassAutobuildInterface):

    def __init__(self, cutoff: float):
        self.cutoff = cutoff

    def __call__(self, event: EventInterface, autobuild_result: AutobuildResultInterface) -> bool:
        selected_event = autobuild_result.selected_fragment_path
        selected_event_score = autobuild_result.scores[selected_event]
        if selected_event_score > self.cutoff:
            return True
        else:
            return False
