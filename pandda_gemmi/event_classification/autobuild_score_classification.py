from pandda_gemmi.analyse_interface import *


class GetEventClassAutobuildScore(GetEventClassAutobuildInterface):

    def __init__(self, cutoff_inspection: float, cutoff_hit: float):
        self.cutoff_inspection = cutoff_inspection
        self.cutoff_hit = cutoff_hit
        self.tag: Literal["autobuild"] = "autobuild"

    def __call__(self, event: EventInterface, autobuild_result: AutobuildResultInterface) -> EventClasses:
        selected_event = autobuild_result.selected_fragment_path
        if not selected_event:
            return EventClasses.JUNK

        selected_event_score = autobuild_result.scores[selected_event]
        if selected_event_score > self.cutoff_hit:
            return EventClasses.HIT
        elif selected_event_score > self.cutoff_inspection:
            return EventClasses.NEEDS_INSPECTION
        else:
            return EventClasses.JUNK
