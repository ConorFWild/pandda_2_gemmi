import os
import dataclasses

from pandda_gemmi.analyse_interface import *


class RescoreEventsAutobuildScore:
    def __call__(self,
                 event_id: EventIDInterface,
                 event: EventInterface,
                 pandda_fs_model: PanDDAFSModelInterface,
                 dataset: DatasetInterface,
                 event_score: float,
                 autobuild_result: AutobuildResultInterface,
                 grid: GridInterface,
                 debug: Debug = Debug.DEFAULT,
                 ):
        selected_event = autobuild_result.selected_fragment_path
        if not selected_event:
            return -0.01

        selected_event_score = autobuild_result.scores[selected_event]
        return selected_event_score
