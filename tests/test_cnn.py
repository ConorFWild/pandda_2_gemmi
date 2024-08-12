from pathlib import Path

import gemmi

from pandda_gemmi.cnn import BuildScorer, LitBuildScoring, load_model_from_checkpoint


def test_BuildScorer():
    data_dir = Path('data')
    model_path = data_dir / 'model_build.ckpt'
    xmap_path = data_dir / 'xmap.ccp4'
    zmap_path = data_dir / 'zmap.ccp4'
    build_good_path = data_dir / 'build_good.pdb'
    build_bad_path = data_dir / 'build_bad.pdb'

    # Get the build model
    model = load_model_from_checkpoint(model_path, LitBuildScoring()).test()

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
