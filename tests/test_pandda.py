import multiprocessing

import pytest
import pandas as pd

from pandda_gemmi.args import PanDDAArgs
from pandda_gemmi.pandda.pandda import pandda
from pandda_gemmi import constants

def test_pandda(test_data, integration_test_out_dir):
    """
    Tests whether PanDDA runs to completion correctly
    :param test_data:
    :param integration_test_out_dir:
    :return:
    """

    args= PanDDAArgs(
        test_data,
        integration_test_out_dir,
        local_cpus=multiprocessing.cpu_count()
    )
    pandda(args)

    pandda_analysis_path = integration_test_out_dir / constants.PANDDA_ANALYSES_DIR
    assert pandda_analysis_path.exists()

    pandda_analyse_events_path = pandda_analysis_path / constants.PANDDA_ANALYSE_EVENTS_FILE
    assert pandda_analyse_events_path.exists()

@pytest.mark.order(after="test_pandda")
def test_reproduce_known_hit_events(integration_test_out_dir):
    pandda_analysis_path = integration_test_out_dir / constants.PANDDA_ANALYSES_DIR
    pandda_analyse_events_path = pandda_analysis_path / constants.PANDDA_ANALYSE_EVENTS_FILE

    # Get the event table
    event_table = pd.read_csv(pandda_analyse_events_path)
    assert len(event_table) > 1

    # Go through the known hit events, and check whether there is a new event close, warning if not


    ...

