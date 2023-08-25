import os
import sys
import subprocess
from pathlib import Path

import streamlit as st

st.title("Run PanDDA 2")


def is_int(num):
    try:
        num = int(num)
        return True
    except Exception:
        return False


with st.form("Run PanDDA 2"):
    data_dir = st.text_input(
        "Directory of PanDDA 2 input data:",
    )

    out_dir = st.text_input(
        "Directory of PanDDA 2 output:",
    )

    num_cpus = st.text_input(
        "Number of cpus to use:",
    )

    submission_options = ["qsub", "local"]
    submission = st.selectbox("How to run job", submission_options)

    submit = st.form_submit_button()

if submit:
    can_run = True

    data_dir_path = Path(data_dir)
    if data_dir_path.exists():
        st.write("Data directory found!")
    else:
        st.write("Data directory not found!")
        can_run = False

    out_dir_path = Path(out_dir)

    if is_int(num_cpus):
        st.write(f'Number of cpus to use is: {num_cpus}')
    else:
        st.write(f"Number of cpus to use is not an integer!")
        can_run = False

    if can_run:
        if not out_dir_path.exists():
            os.mkdir(out_dir_path)

        submit_script_path = out_dir_path / "submit.sh"
        script_directory = Path(os.path.dirname(os.path.abspath(sys.argv[0])))
        pandda_script_path = script_directory / 'pandda.py'

        if submission == "qsub":

            with open(submit_script_path, "w") as f:
                f.write(
                    "#!/bin/sh"
                    f"python -u {pandda_script_path} --data_dirs={data_dir_path} --out_dir={out_dir_path} --local_cpus={num_cpus}"

                )
            p = subprocess.Popen(
                f"qsub -V -pe smp {num_cpus} -l m_mem_free={int(180 / num_cpus)}G {submit_script_path}",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = p.communicate()
            st.write("Stdout")
            st.write(stdout)
            st.write("Stderr")
            st.write(stderr)


        elif submission == "local":
            p = subprocess.Popen(
                f"python -u {pandda_script_path} --data_dirs={data_dir_path} --out_dir={out_dir_path} --local_cpus={num_cpus}",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = p.communicate()
            st.write("Stdout")
            st.write(stdout)
            st.write("Stderr")
            st.write(stderr)
            ...
