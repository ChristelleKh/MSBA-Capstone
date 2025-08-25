# Home.py
import streamlit as st
from lib.data import DATA_FILES, DEFAULT_WEIGHTS

st.set_page_config(page_title="FDI Analytics (Multi‑Page)", layout="wide")

st.title("FDI Analytics — Multi‑Page App")
st.write("Use the left sidebar to switch pages.")

st.sidebar.header("Local data files")
for k, v in DATA_FILES.items():
    st.sidebar.write(f"*{k}* → {v}")

st.sidebar.header("Weights (default)")
st.sidebar.json(DEFAULT_WEIGHTS)
