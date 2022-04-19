from typing import *

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
from pandda_gemmi.scoring import EXPERIMENTAL_score_structure_signal_to_noise_density


class GetEventScoreAutobuild(GetEventScoreAutobuildInterface):
    def __call__(self, *args, **kwargs):
        ...