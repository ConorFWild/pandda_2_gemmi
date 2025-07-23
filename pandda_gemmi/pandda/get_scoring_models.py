import os
import shutil
import time
import inspect

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except ImportError:
    print('No sklearn-express available!')

import gdown
import yaml

import numpy as np
import pandas as pd
import gemmi

from pandda_gemmi.interfaces import *
from pandda_gemmi.args import PanDDAArgs
from pandda_gemmi.fs import PanDDAFS
from pandda_gemmi.dataset import XRayDataset, StructureArray, Structure
from pandda_gemmi.dmaps import (
    SparseDMap,
    SparseDMapStream,
    TruncateReflections,
    SmoothReflections,
)
from pandda_gemmi.alignment import Alignment, DFrame
from pandda_gemmi.processor import ProcessLocalRay, Partial
from pandda_gemmi.comparators import (
    get_comparators,
    FilterRFree,
    FilterRange,
    FilterExcludeFromAnalysis,
    FilterOnlyDatasets,
    FilterSpaceGroup,
    FilterResolution,
    FilterCompatibleStructures,
    FilterResolutionLowerLimit,
    FilterNoLigandData
)
from pandda_gemmi.event_model.event import EventBuild
from pandda_gemmi.event_model.characterization import get_characterization_sets, CharacterizationNNAndFirst
from pandda_gemmi.event_model.filter_characterization_sets import filter_characterization_sets
from pandda_gemmi.event_model.outlier import PointwiseNormal, PointwiseMAD
from pandda_gemmi.event_model.cluster import ClusterDensityDBSCAN
from pandda_gemmi.event_model.score import get_model_map, ScoreCNNLigand
from pandda_gemmi.event_model.filter import (
    FilterSize,
    FilterScore,
    FilterSymmetryPosBuilds,
    FilterLocallyHighestBuildScoring
)
from pandda_gemmi.event_model.select import select_model
from pandda_gemmi.event_model.output import output_maps
from pandda_gemmi.event_model.filter_selected_events import filter_selected_events
from pandda_gemmi.event_model.get_bdc import get_bdc

from pandda_gemmi.site_model import HeirarchicalSiteModel, Site, get_sites
from pandda_gemmi.autobuild import AutobuildResult, ScoreCNNEventBuild
from pandda_gemmi.autobuild.inbuilt import mask_dmap, get_conformers, autobuild_conformer
from pandda_gemmi.autobuild.merge import merge_autobuilds, MergeHighestBuildScore
from pandda_gemmi.ranking import rank_events, RankHighEventScore, RankHighEventScoreBySite
from pandda_gemmi.tables import output_tables
from pandda_gemmi.pandda_logging import PanDDAConsole
from pandda_gemmi import serialize
from pandda_gemmi.cnn import load_model_from_checkpoint, EventScorer, LitEventScoring, BuildScorer, LitBuildScoring, \
    set_structure_mean

from pandda_gemmi.metrics import get_hit_in_site_probabilities
from pandda_gemmi.plots import plot_aligned_density_projection
from pandda_gemmi.pandda.process_dataset import process_dataset
from pandda_gemmi.pandda.process_model import ProcessModel


def get_scoring_models(args, ):
    # Get the method for scoring events
    if args.use_ligand_data:
        event_model_path = Path(os.path.dirname(inspect.getfile(LitEventScoring))) / "model_event.ckpt"
        event_config_path = Path(os.path.dirname(inspect.getfile(LitEventScoring))) / "model_event_config.yaml"
        event_score_quantiles_path = Path(
            os.path.dirname(inspect.getfile(LitEventScoring))) / "event_score_quantiles.csv"
        if not (event_model_path.exists() & event_config_path.exists()):
            print(f'No event model at {event_model_path}. Downloading event model...')
            with open(event_model_path, 'wb') as f:
                # gdown.download('https://drive.google.com/file/d/1b58MUIJdIYyYHr-UhASVCvIWtIgrLYtV/view?usp=sharing',
                #                f)
                gdown.download(id='1b58MUIJdIYyYHr-UhASVCvIWtIgrLYtV',
                               output=f)
            with open(event_config_path, 'wb') as f:
                gdown.download(id='1qyPqPylOguzXmt6XSFaXCKrnvb8gZ8E2',
                               output=f)
            with open(event_score_quantiles_path, 'wb') as f:
                gdown.download(id='15RnkrGtEmFvtBvIlwfaUE1QfQrD2npnu', output=f)

        with open(event_config_path, 'r') as f:
            event_model_config = yaml.safe_load(f)
        score_event_model = load_model_from_checkpoint(
            event_model_path,
            LitEventScoring(event_model_config),
        ).float().eval()
        score_event = EventScorer(score_event_model, event_model_config, debug=args.debug)

        # Get the method for scoring
        build_model_path = Path(os.path.dirname(inspect.getfile(LitBuildScoring))) / "model_build.ckpt"
        build_config_path = Path(os.path.dirname(inspect.getfile(LitBuildScoring))) / "model_build_config.yaml"

        if not (build_model_path.exists() & build_config_path.exists()):
            print(f'No build model at {build_model_path}.Downloading build model...')
            with open(build_model_path, 'wb') as f:
                # gdown.download('https://drive.google.com/file/d/17ow_rxuEvi0LitMP_jTWGMSDt-FfJCkR/view?usp=sharing',
                #                f
                #                )
                gdown.download(id='17ow_rxuEvi0LitMP_jTWGMSDt-FfJCkR',
                               output=f)
            with open(build_config_path, 'wb') as f:
                gdown.download(id='1HEXHZ6kfh92lQoWBHalGbUJ-iCsOIkFo',
                               output=f)
        with open(build_config_path, 'r') as f:
            build_model_config = yaml.safe_load(f)
        score_build_model = load_model_from_checkpoint(
            build_model_path,
            LitBuildScoring(build_model_config),
        ).float().eval()
        score_build = BuildScorer(score_build_model, build_model_config)
        event_score_quantiles = pd.read_csv(event_score_quantiles_path)
    else:  # use_ligand_data=False
        event_model_path = Path(os.path.dirname(inspect.getfile(LitEventScoring))) / "model_event_no_ligand.ckpt"
        event_config_path = Path(
            os.path.dirname(inspect.getfile(LitEventScoring))) / "model_event_no_ligand_config.yaml"
        event_score_quantiles_path = Path(
            os.path.dirname(inspect.getfile(LitEventScoring))) / "event_score_no_ligand_quantiles.csv"
        if not (event_model_path.exists() & event_config_path.exists()):
            print(f'No event model at {event_model_path}. Downloading event model...')
            with open(event_model_path, 'wb') as f:
                gdown.download(id='1ccUM3g6RKluxwz8hofqmXEH2iymMvjyy',
                               output=f)
            with open(event_config_path, 'wb') as f:
                gdown.download(id='1c_QyEjFD5DtYlbU-Gh1o79gkbtdSrkDl',
                               output=f)
            with open(event_score_quantiles_path, 'wb') as f:
                gdown.download(id='1kHtBtLgGBuSBO8Mrf9pn7kjokL6fRMP6', output=f)

        with open(event_config_path, 'r') as f:
            event_model_config = yaml.safe_load(f)
        score_event_model = load_model_from_checkpoint(
            event_model_path,
            LitEventScoring(event_model_config),
        ).float().eval()
        score_event = EventScorer(score_event_model, event_model_config, debug=args.debug)
        event_score_quantiles = pd.read_csv(event_score_quantiles_path)
        score_build = None
    if args.debug:
        print(f'Using ligand?: {score_event.model.ligand} / {score_event.model.ligand is True}')
        print(f'Score model path: {event_model_path}')

    return score_event, score_build, event_model_config, event_score_quantiles
