from pathlib import Path
import argparse

import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt


@st.cache_data
def get_table(pandda_inspect_events):
    return pd.read_csv(pandda_inspect_events)

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("pandda_dir")

    args = parser.parse_args()

    pandda_inspect_events = Path(args.pandda_dir) / "analyses" / "pandda_inspect_events.csv"

    st.title("PanDDA Inspect Table")

    st.write(pandda_inspect_events)

    table = get_table(pandda_inspect_events)
    st.line_chart(data=table, x=None, y="z_peak", )

    fig = plt.figure()
    plt.scatter(x=table["z_peak"], y=table["cluster_size"], s=0.1)
    plt.yscale("log")
    st.pyplot(fig)

    events = {}
    for _idx, _row in table.iterrows():
        dtag, event_idx = _row["dtag"], _row["event_idx"]
        events[(dtag, event_idx)] = _row
    option = st.selectbox("", [x for x in events])

    if option is not None:
        st.write(f"{option[0]} : {option[1]}")
        st.write(events[option])

    st.write(table)




if __name__ == "__main__":
    __main__()
