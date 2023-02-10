import os
from pathlib import Path

import pickle
import secrets


def cache(directory: Path, object):
    code = secrets.token_hex(16)

    path = directory / f"cache_{code}.pickle"
    with open(path, "wb") as f:
        pickle.dump(object, f)

    return path


def uncache(path: Path, remove: bool = False):
    with open(path, "rb") as f:
        object = pickle.load(f)

    if remove:
        os.remove(path)

    return object
