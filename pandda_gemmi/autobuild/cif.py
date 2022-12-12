from __future__ import annotations

import os
import dataclasses
import subprocess
from pathlib import Path
import json

from typing import *

import fire
import numpy as np
import gemmi
# import ray

from pandda_gemmi.analyse_interface import *
from pandda_gemmi import constants


def execute(command: str):
    p = subprocess.Popen(command,
                         shell=True,
                         env=os.environ.copy(),
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         )

    stdout, stderr = p.communicate()

    print(stdout, stderr)


# #####################
# # Generate cif
# #####################

def get_elbow_command(smiles_file: Path, out_dir: Path) -> str:
    command = constants.ELBOW_COMMAND.format(
        out_dir=str(out_dir),
        smiles_file=str(smiles_file),
        prefix=constants.LIGAND_PREFIX, )
    return command


def generate_cif(smiles_path: Path, out_dir: Path):
    # Get the command to run elbow
    elbow_command = get_elbow_command(smiles_path, out_dir)

    # Run the command
    execute(elbow_command)

    return out_dir / constants.LIGAND_CIF_FILE, out_dir / constants.LIGAND_PDB_FILE


def get_grade_command(smiles_file: Path, out_dir: Path) -> str:
    command = constants.GRADE_COMMAND.format(
        out_dir=str(out_dir.resolve()),
        smiles_file=str(smiles_file.resolve()),
        prefix=constants.LIGAND_PREFIX, )
    return command


def generate_cif_grade(smiles_path: Path, out_dir: Path):
    # Get the command to run elbow
    grade_command = get_grade_command(smiles_path, out_dir)

    # Run the command
    execute(grade_command)

    return out_dir / constants.LIGAND_CIF_FILE


def get_grade2_command(smiles_file: Path, out_dir: Path) -> str:
    command = constants.GRADE2_COMMAND.format(
        out_dir=str(out_dir),
        smiles_file=str(smiles_file),
        prefix=constants.LIGAND_PREFIX, )
    return command


def generate_cif_grade2(smiles_path: Path, out_dir: Path):
    # Get the command to run elbow
    grade_2_command = get_grade2_command(smiles_path, out_dir)

    # Run the command
    execute(grade_2_command)

    return out_dir / constants.LIGAND_CIF_FILE
