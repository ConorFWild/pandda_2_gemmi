# 3rd party
import numpy as np
import gemmi
from rdkit import Chem
from rdkit.Chem import AllChem
import joblib
import scipy
from scipy import spatial as spsp, optimize
from pathlib import Path
import time

#
from pandda_gemmi.analyse_interface import *
from pandda_gemmi.dataset import Dataset
# from pandda_gemmi.fs import PanDDAFSModel, ProcessedDataset
from pandda_gemmi.event import Cluster
# from pandda_gemmi.autobuild import score_structure_signal_to_noise_density, EXPERIMENTAL_score_structure_signal_to_noise_density
from pandda_gemmi.scoring import EXPERIMENTAL_score_structure_signal_to_noise_density, score_structure_contour
from pandda_gemmi.python_types import *


class ConfomerID(ConfromerIDInterface):
    def __init__(self, conformer_id):
        self.conformer_id = conformer_id

    def __int__(self):
        return self.conformer_id

    def __hash__(self):
        return self.conformer_id


class Conformers(ConformersInterface):

    def __init__(self,
                 conformers: Dict[ConfromerIDInterface, Any],
                 method: str,
                 path: Optional[Path],
                 ):
        self.conformers: Dict[ConfromerIDInterface, Any] = conformers
        self.method: str = method
        self.path: Optional[Path] = path


class ConformerFittingResult(ConformerFittingResultInterface):
    def __init__(self,
                 score: Optional[float],
                 optimised_fit: Optional[Any],
                 score_log: Optional[Dict]
                 ):
        self.score: Optional[float] = score
        self.optimised_fit: Optional[Any] = optimised_fit
        self.score_log = score_log

    def log(self) -> Dict:
        return {
            "Score": str(self.score),
            "Score Log": self.score_log
        }


class LigandFittingResult(LigandFittingResultInterface):
    def __init__(self,
                 conformer_fitting_results: ConformerFittingResultsInterface,
                 conformers: ConformersInterface,
                 # selected_conformer: int
                 ):
        self.conformer_fitting_results: ConformerFittingResultsInterface = conformer_fitting_results
        # selected_conformer: int = selected_conformer
        self.conformers = conformers

    def log(self) -> Dict:
        return {
            "Conformer results": {
                str(conformer_id.conformer_id): conformer_result.log()
                for conformer_id, conformer_result
                in self.conformer_fitting_results.items()
            },
            "Conformer generation info": self.conformers.log()
        }


class EventScoringSizeResult(EventScoringResultInterface):
    def __init__(self, ligand_fitting_result, event):
        # self.score: float = score
        self.ligand_fitting_result: LigandFittingResultInterface = ligand_fitting_result
        self.event: EventInterface = event

    def get_selected_structure(self) -> Any:
        return None

    def get_selected_structure_score(self) -> Optional[float]:
        return float(self.event.cluster.values.size)

    def log(self) -> Dict:
        selected_conformer_key = None
        return {
            "Selected conformer id": str(selected_conformer_key),
            "Selected conformer score": str(self.get_selected_structure_score()),
            "Ligand fitting log": self.ligand_fitting_result.log()
        }


class GetEventScoreSize(GetEventScoreSizeInterface):
    tag: Literal["size"] = "size"

    def __call__(self, events: EventsInterface) -> EventScoringResultsInterface:
        event_scores = {
            event_id: EventScoringSizeResult(
                LigandFittingResult(
                    {
                        ConfomerID(0): ConformerFittingResult(
                            None,
                            None,
                            None
                        )
                    },
                    Conformers(
                        {},
                        "None",
                        None,
                    ),
                ),
                event,
            )
            for event_id, event
            in events.events.items()
        }

        return event_scores
