import streamlit as st

st.title("Run PanDDA 2")

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

    st.form_submit_button()

