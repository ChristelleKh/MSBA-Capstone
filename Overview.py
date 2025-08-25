# pages/01_Overview.py
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------- Settings ----------
DATA_FILE = "world_bank_data_with_scores_and_continent.csv"  # local file in repo root

# ---------- Small helpers ----------
def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # drop duplicate-named columns (pyarrow/streamlit hates dups)
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def guess_cols(df: pd.DataFrame):
    cols = df.columns.tolist()
    # country column
    for c in ["country", "Country", "Country Name", "Country_Name", "Economy"]:
        if c in cols:
            country_col = c
            break
    else:
        country_col = cols[0]  # fallback to first column
    # year column
    for y in ["year", "Year", "YEAR", "yr"]:
        if y in cols:
            year_col = y
            break
    else:
        # if no obvious year col, create a placeholder to avoid crashes
        year_col = None
    return country_col, year_col

# ---------- UI ----------
st.title("Overview")
st.caption("Quick KPIs and a world map (if composite scores are available).")

# Load data (local only; no uploads, no web)
try:
    raw = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    st.warning(f"Place `{DATA_FILE}` in the repo root to view the overview.")
    st.stop()

df = ensure_columns(raw)
country_col, year_col = guess_cols(df)

# KPIs
c1, c2, c3 = st.columns(3)
try:
    n_countries = df[country_col].nunique()
except Exception:
    n_countries = len(df)
c1.metric("Countries", f"{n_countries:,}")

if year_col and year_col in df.columns:
    years = (
        pd.Series(df[year_col])
        .dropna()
        .astype(int, errors="ignore")
        .unique()
        .tolist()
    )
    years = sorted([int(y) for y in years if pd.notna(y)])
    c2.metric("Years", f"{len(years)}")
    latest_year = years[-1] if years else None
    c3.metric("Latest year", latest_year if latest_year is not None else "—")
else:
    years, latest_year = [], None
    c2.metric("Years", "—")
    c3.metric("Latest year", "—")

st.markdown("---")

# World map (only if composite_score exists)
if "composite_score" in df.columns and country_col:
    # year selector (falls back to latest if available, else shows all)
    if years:
        sel_year = st.slider(
            "Year",
            int(min(years)), int(max(years)),
            int(latest_year) if latest_year is not None else int(max(years))
        )
        plot_df = df[df[year_col] == sel_year].copy()
        title_suffix = f"— {sel_year}"
    else:
        plot_df = df.copy()
        title_suffix = ""

    if plot_df.empty:
        st.info("No rows for the selected year.")
    else:
        fig = px.choropleth(
            plot_df,
            locations=country_col,
            locationmode="country names",
            color="composite_score",
            hover_name=country_col,
            color_continuous_scale="Viridis",
            range_color=(0, 1),
            title=f"Composite Score {title_suffix}".strip(),
        )
        fig.update_layout(coloraxis_colorbar_title="Score (0–1)")
        st.plotly_chart(fig, use_container_width=True)

        # small preview table
        with st.expander("Preview (top 25 by score)"):
            show_cols = [c for c in [country_col, "composite_score", "grade"] if c in plot_df.columns]
            st.dataframe(
                plot_df.sort_values("composite_score", ascending=False)[show_cols].head(25),
                use_container_width=True
            )
else:
    st.info(
        "No `composite_score` found in the dataset. "
        "Once you compute scores (on the Scoring page), this map will appear here."
    )

# Optional raw preview
with st.expander("Raw data preview"):
    st.dataframe(df.head(30), use_container_width=True)
