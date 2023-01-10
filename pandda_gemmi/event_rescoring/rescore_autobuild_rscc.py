import os
import dataclasses

from pandda_gemmi.event_rescoring.get_rscc_phenix import get_rscc
from pandda_gemmi.analyse_interface import *


class RescoreEventsAutobuildRSCC:
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
        dataset_bound_state_model_path = autobuild_result.selected_fragment_path
        event_map_path = pandda_fs_model.processed_datasets.processed_datasets[event_id.dtag].event_map_files[
            event_id.event_idx].path
        resolution = dataset.reflections.get_resolution()
        tmp_dir = pandda_fs_model.processed_datasets.processed_datasets[event_id.dtag].path / f"{event_id.event_idx.event_idx}"

        # Make the tmp dir if there is one
        if not tmp_dir.exists():
            os.mkdir(tmp_dir)

        # Get RSCCs from phenix
        rsccs: Optional[RSCCSInterface] = get_rscc(
            dataset_bound_state_model_path,
            event_map_path,
            resolution,
            tmp_dir,
        )

        # Get an RSCC if there is one
        if not rsccs:
            return -0.01
        if len(rsccs) == 0:
            return -0.01


        return max(rsccs.values())
