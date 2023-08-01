from pathlib import Path
import argparse

import streamlit as st
import pandas as pd


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("pandda_dir")

    args = parser.parse_args()

    pandda_inspect_events = Path(args.pandda_dir) / "analyses" / "pandda_inspect_events.csv"

    st.title("PanDDA Inspect Table")

    st.write(pandda_inspect_events)

    st.write(pd.read_csv(pandda_inspect_events))


if __name__ == "__main__":
    __main__()