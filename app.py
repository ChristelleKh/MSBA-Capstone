# app.py  (acts as Home)
import os
import streamlit as st

st.set_page_config(page_title="FDI Analytics (Multi‑Page)", layout="wide")

st.title("FDI Analytics — Home")
st.write("Use the left sidebar to open **Overview**.")

st.sidebar.header("Navigation")
st.sidebar.write("Pages are in the `pages/` folder. Click **Overview** to start.")

# Optional: quick sanity check that your data file exists
DATA_FILE = "world_bank_data_with_scores_and_continent.csv"
if os.path.exists(DATA_FILE):
    st.sidebar.success(f"Found `{DATA_FILE}`")
else:
    st.sidebar.warning(f"Put `{DATA_FILE}` in the repo root so the Overview page can load it.")
