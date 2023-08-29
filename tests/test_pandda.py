import multiprocessing

import pytest

from pandda_gemmi.args import PanDDAArgs
from pandda_gemmi.pandda.pandda import pandda

def test_pandda(test_data, integration_test_out_dir):

    args= PanDDAArgs(
        test_data,
        integration_test_out_dir,
        local_cpus=multiprocessing.cpu_count()
    )
    pandda(args)


