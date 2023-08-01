from pathlib import Path
import argparse

import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("pandda_dir")

    args = parser.parse_args()

    pandda_inspect_events = Path(args.pandda_dir) / "analyses" / "pandda_inspect_events.csv"

    st.title("PanDDA Inspect Table")

    st.write(pandda_inspect_events)



    table = pd.read_csv(pandda_inspect_events)
    st.line_chart(data=table, x=None, y="z_peak")

    fig = plt.scatter(x=table["z_peak"], y=table["cluster_size"])
    st.pyplot(fig)

    st.write(table)


if __name__ == "__main__":
    __main__()