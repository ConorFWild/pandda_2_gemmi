import os
from pathlib import Path

import fire


def rename(directory: str, dry=False):
    directory = Path(directory)

    for path in directory.rglob("*"):
        path = path.resolve()

        if path.suffix == ".pdb":

            new_path = path.parent / f"{path.parent.name}.pdb"
            if dry:
                print(f"Would rename {path} to {new_path}")
            else:
                path.rename(new_path)

        elif path.suffix == ".mtz":
            new_path = path.parent / f"{path.parent.name}.mtz"

            if dry:
                print(f"Would rename {path} to {new_path}")
            else:
                path.rename(new_path)

        else:
            continue


if __name__ == "__main__":
    fire.Fire(rename)
