import os
from typing import *
from pathlib import Path
from dataclasses import dataclass
import re

import fire

from fragalysis_api.xcextracter.xcextracter import xcextracter
from fragalysis_api.xcextracter.getdata import GetTargetsData, GetPdbData

# FUnctions for getting fragalysis data
def fetch_data(fragalsis_id: str, results_dir: Path):
    summary = xcextracter(fragalsis_id)

    for index, row in summary.iterrows():
        prot_id = row["protein_code"]
        pdb_grabber = GetPdbData()

        print(prot_id)
        match = re.match(r"(([^_]+)_([^:]+))", prot_id)
        print(match)
        print(match.groups())
        all, dtag, num = match.groups()

        # use our selected code to pull out the pdb file (currently the file with ligand removed)
        print(prot_id)
        try:
            print("Got bound pdb block")
            if not results_dir.exists():
                os.mkdir(results_dir)

            dtag_dir = results_dir / dtag
            if not dtag_dir.exists():
                os.mkdir(dtag_dir)

            pdb_file = dtag_dir / num
            if not pdb_file.exists():
                pdb_block = pdb_grabber.get_bound_pdb_file(prot_id)

                with open(pdb_file, "w") as f:
                    f.write(pdb_block)

        except Exception as e:
            print(e)


# Functions for getting RMSD
@dataclass()
class RMSDResult:
    ...


def rmsd_from_structures(structure_1, structure_2) -> RMSDResult:
    ...


def structure_from_path():
    ...


def model_path_from_pandda_dir(pandda_dir, dtag):
    ...


# Functions for graphing
def figure_from_results():
    ...


def plot_results(out_path, results):
    figure = figure_from_results(results)

    figure.savefig(str(out_path))


def autobuild_vs_fragalysis(pandda_dir: str, results_dir: str, fragalysis_id: str, ):
    # Type input
    pandda_dir = Path(pandda_dir)
    results_dir = Path(results_dir)

    # Fetch data from fragalysis
    model_paths: Dict[str, Path] = fetch_data(fragalysis_id, results_dir)

    # Loop over comparing
    results: Dict[str, RMSDResult] = {
        dtag: rmsd_from_structures(
            model_paths[dtag],
            model_path_from_pandda_dir(pandda_dir, dtag),
        )
        for dtag, path
        in model_paths.items()
    }

    # Make a graph of results
    plot_results(
        results_dir / "",
        results,
    )


if __name__ == "__main__":
    fire.Fire(autobuild_vs_fragalysis)
