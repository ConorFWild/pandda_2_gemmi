from pathlib import Path

from rich import print as rprint
import gemmi
import numpy as np

from pandda_gemmi.cnn import BuildScorer, LitBuildScoring, load_model_from_checkpoint, EventScorer, LitEventScoring, Event
from pandda_gemmi.fs.pandda_input import LigandFiles
from pandda_gemmi.autobuild.inbuilt import get_conformers

def test_BuildScorer():
    data_dir = Path('data')
    model_path = data_dir / 'model_build.ckpt'
    xmap_path = data_dir / 'xmap.ccp4'
    zmap_path = data_dir / 'zmap.ccp4'
    build_good_path = data_dir / 'build_good.pdb'
    build_bad_path = data_dir / 'build_bad.pdb'

    # Get the build model
    model = load_model_from_checkpoint(model_path, LitBuildScoring()).eval()

    # Get the build scorer
    build_scorer = BuildScorer(model)

    # Get the xmap
    xmap = gemmi.read_ccp4_map(str(xmap_path)).grid

    # Get the zmap
    zmap = gemmi.read_ccp4_map(str(zmap_path)).grid

    # Get the good build path
    good_build = gemmi.read_structure(str(build_good_path))

    # Get the bad build path
    bad_build = gemmi.read_structure(str(build_bad_path))

    # Run the scorer on the good build
    good_build_score = build_scorer(good_build, zmap, xmap)

    # Run the scorer on the bad build
    bad_build_score = build_scorer(bad_build, zmap, xmap)

    # Assert the good score is better than bad one
    assert good_build_score > bad_build_score

def test_EventScorer():
    data_dir = Path('data')
    model_path = data_dir / 'model_event.ckpt'
    xmap_path = data_dir / 'xmap.ccp4'
    zmap_path = data_dir / 'zmap.ccp4'
    ligand_path = data_dir / 'ligand.cif'

    # Get the build model
    model = load_model_from_checkpoint(model_path, LitEventScoring()).eval()

    # Get the build scorer
    event_scorer = EventScorer(model)

    # Get the xmap
    xmap = gemmi.read_ccp4_map(str(xmap_path)).grid

    # Get the zmap
    zmap = gemmi.read_ccp4_map(str(zmap_path)).grid

    # Get the good build path
    good_event = Event(np.array([13.56, 42.74, 31.44]))

    # Get the bad build path
    bad_event = Event(np.array([20.05, 46.52, 20.25]))

    # Get a ligand conf
    conf = get_conformers(LigandFiles(ligand_path, None, None))[0]

    # Run the scorer on the good build
    good_event_score = event_scorer(good_event, conf, zmap, xmap)

    # Run the scorer on the bad build
    bad_event_score = event_scorer(bad_event, conf, zmap, xmap)

    # Assert the good score is better than bad one
    assert good_event_score > bad_event_score