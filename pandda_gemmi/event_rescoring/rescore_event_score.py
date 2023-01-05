import os
import dataclasses

from pandda_gemmi.analyse_interface import *


class RescoreEventsEventScore:
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
        return event_score
