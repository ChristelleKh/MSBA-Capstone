import streamlit as st, plotly.express as px, pandas as pd
from lib.data import load_all_local, ensure_columns

st.title("Interactive World Map")

df, *_ = load_all_local()
if "scored" in st.session_state:
    df = st.session_state["scored"]

if df is None:
    st.info("Compute scores first or load the panel CSV.")
    st.stop()

df = ensure_columns(df)
country_col = "country" if "country" in df.columns else df.columns[0]
year_col = "year" if "year" in df.columns else "Year"

years = sorted(pd.Series(df[year_col]).dropna().unique())
sel_year = st.slider("Year", int(min(years)), int(max(years)), int(max(years)))
conts = ["All"]
if "continent" in df.columns:
    conts += sorted([c for c in df["continent"].dropna().unique()])
sel_cont = st.selectbox("Continent", conts, index=0)

plot_df = df[df[year_col]==sel_year].copy()
if sel_cont != "All" and "continent" in plot_df.columns:
    plot_df = plot_df[plot_df["continent"]==sel_cont]

if "composite_score" not in plot_df.columns:
    st.info("No composite scores yet. Use the *Scoring* page.")
else:
    fig = px.choropleth(
        plot_df,
        locations=country_col,
        locationmode="country names",
        color="composite_score",
        hover_name=country_col,
        color_continuous_scale="Viridis",
        range_color=(0,1),
        title=f"Composite Score — {sel_year}" + (f" — {sel_cont}" if sel_cont!='All' else "")
    )
    fig.update_layout(coloraxis_colorbar_title="Score (0–1)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("*Drill‑through: time series*")
    c_list = sorted(plot_df[country_col].unique())
    if c_list:
        chosen = st.selectbox("Country", c_list)
        long = df[df[country_col]==chosen].sort_values
