import os

from .pandda_fs import PanDDAFS
from .pandda_input import PanDDAInput
from .pandda_output import PanDDAOutput


def try_make(path):
    try:
        os.mkdir(path)
    except:
        return