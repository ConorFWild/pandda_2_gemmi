import shutil
from pathlib import Path

import pytest

from pandda_gemmi import constants

@pytest.fixture(scope="session")
def test_data():
    test_data_path = Path(constants.TEST_DATA_DIR)
    if test_data_path.exists():
        return test_data_path
    else:
        # TODO: Get
        return None

@pytest.fixture(scope="session")
def integration_test_out_dir():
    path = Path('test')
    if path.exists():
        shutil.rmtree(path)

    return path
