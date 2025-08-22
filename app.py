import os
import io
import json
import math
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

# -----------------------------------------------------------------------------
# CONFIG: point these to your repo/branch/folder holding the data files
# Example: https://raw.githubusercontent.com/<user>/<repo>/<branch>/data/
# -----------------------------------------------------------------------------
RAW_BASE = st.secrets.get(
    "RAW_BASE",
    "https://raw.githubusercontent.com/<YOUR_USER>/<YOUR_REPO>/<YOUR_BRANCH>/",  # <-- CHANGE ME
)

# File names exactly as you told me
FILES = {
    "world_bank": "world_bank_data_with_scores_and_continent.csv",
    "sectors": "merged_sectors_data.csv",
    "destinations": "merged_destinations_data.csv",
    "capex_eda": "capex_EDA.xlsx",   # Excel
}

# Indicator weights (sum to 100). Positive direction unless noted.
# If a column is missing, it’s ignored gracefully.
INDICATOR_WEIGHTS = {
    # Economic
    "GDP growth (annual %)": 10,
    "GDP per capita, PPP (current international $)": 8,
    "Current account balance (% of GDP)": 6,
    "Foreign direct investment, net outflows (% of GDP)": 6,
    "Inflation, consumer prices (annual %)": {"weight": 5, "direction": "negative"},
    "Exports of goods and services (% of GDP)": 5,
    "Imports of goods and services (% of GDP)": 5,
    # Governance
    "Political Stability and Absence of Violence/Terrorism: Estimate": 12,
    "Government Effectiveness: Estimate": 10,
    "Control of Corruption: Estimate": 8,
    # Infrastructure/Readiness
    "Access to electricity (% of population)": 9,
    "Individuals using the Internet (% of population)": 8,
    "Total reserves in months of imports": 8,
}

CATEGORY_BREAKDOWN = {
    "Economic Fundamentals": [
        "GDP growth (annual %)",
        "GDP per capita, PPP (current international $)",
        "Current account balance (% of GDP)",
        "Foreign direct investment, net outflows (% of GDP)",
        "Inflation, consumer prices (annual %)",
        "Exports of goods and services (% of GDP)",
        "Imports of goods and services (% of GDP)",
    ],
    "Governance & Institutions": [
        "Political Stability and Absence of Violence/Terrorism: Estimate",
        "Government Effectiveness: Estimate",
        "Control of Corruption: Estimate",
    ],
    "Infrastructure & Readiness": [
        "Access to electricity (% of population)",
        "Individuals using the Internet (% of population)",
        "Total reserves in months of imports",
    ],
}

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def raw_url(path: str) -> str:
    base = RAW_BASE.rstrip("/")
    return f"{base}/{path.lstrip('/')}"

@st.cache_data(show_spinner=False, ttl=60 * 10)
def read_csv_from_github(path: str) -> pd.DataFrame:
    url = raw_url(path)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))

@st.cache_data(show_spinner=False, ttl=60 * 10)
def read_excel_from_github(path: str, sheet_name=0) -> pd.DataFrame:
    url = raw_url(path)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    try:
        return pd.read_excel(io.BytesIO(r.content), sheet_name=sheet_name, engine="openpyxl")
    except Exception as e:
        raise RuntimeError(
            "Reading Excel failed. Make sure `openpyxl` is in requirements.txt "
            "and the file path/extension is correct."
        ) from e

def col_present(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns

def minmax_series(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mask = s.notna()
    if mask.sum() <= 1:
        return pd.Series(np.nan, index=s.index)
    lo, hi = s[mask].min(), s[mask].max()
    if hi == lo:
        return pd.Series(0.5, index=s.index)  # avoid 0 division, neutral
    out = (s - lo) / (hi - lo)
    return out

def compute_scores(df: pd.DataFrame, year_col="Year", country_col="Country"):
    """
    If df already has 'Score' and 'Grade', we use them.
    Otherwise we build them from the indicator weights above.
    """
    df = df.copy()
    # Name harmonization best-effort
    for c in [year_col, country_col]:
        if c not in df.columns:
            # try common fallbacks
            candidates = {
                "Year": ["year", "Year ", "year_"],
                "Country": ["country", "Country Name", "Country_Name", "CountryName"],
            }[c]
            for alt in candidates:
                if alt in df.columns:
                    df.rename(columns={alt: c}, inplace=True)
                    break

    # if Score exists, just return after basic cleaning
    if "Score" in df.columns:
        # Still ensure Grade present
        if "Grade" not in df.columns:
            df["Grade"] = grade_by_year(df, "Score", year_col=year_col)
        return df

    # Otherwise compute from indicators
    weight_entries = []
    for key, meta in INDICATOR_WEIGHTS.items():
        w = meta["weight"] if isinstance(meta, dict) else meta
        direction = meta.get("direction", "positive") if isinstance(meta, dict) else "positive"
        if key in df.columns:
            norm = minmax_series(df[key])
            if direction == "negative":  # invert
                norm = 1 - norm
            weight_entries.append((key, w, norm))
    if not weight_entries:
        st.warning("No matching indicator columns found to compute a score. "
                   "Will return original dataframe.")
        return df

    # normalize weights to sum to 1 (only for the indicators that exist)
    weights = np.array([w for _, w, _ in weight_entries], dtype=float)
    weights = weights / weights.sum()

    # weighted sum
    score = np.zeros(len(df), dtype=float)
    for (_, _, norm), w in zip(weight_entries, weights):
        score += w * norm.fillna(norm.mean())

    df["Score"] = score
    df["Grade"] = grade_by_year(df, "Score", year_col=year_col)
    return df

def grade_by_year(df: pd.DataFrame, score_col: str, year_col="Year"):
    grades = []
    for _, g in df.groupby(df[year_col], dropna=False):
        # percentiles
        p = g[score_col].rank(pct=True)
        g_grade = pd.cut(
            p,
            bins=[0, 0.25, 0.50, 0.75, 0.90, 1.00],
            labels=["D", "C", "B", "A", "A+"],
            include_lowest=True,
            right=True,
        )
        grades.append(g_grade)
    return pd.concat(grades).sort_index()

def safe_number(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=True, ttl=60 * 10)
def load_all():
    wb = read_csv_from_github(FILES["world_bank"])
    wb = compute_scores(wb)

    sectors = read_csv_from_github(FILES["sectors"])
    dest = read_csv_from_github(FILES["destinations"])

    capex = read_excel_from_github(FILES["capex_eda"])  # sheet 0
    # Try to harmonize CAPEX columns: expect Year + CAPEX (or similar)
    # light rename guesses:
    for cand in ["year", "Year ", "YEAR"]:
        if cand in capex.columns and "Year" not in capex.columns:
            capex.rename(columns={cand: "Year"}, inplace=True)
    for cand in ["capex", "CAPEX", "Capex", "Capex ($B)", "CAPEX ($B)"]:
        if cand in capex.columns and "CAPEX" not in capex.columns:
            capex.rename(columns={cand: "CAPEX"}, inplace=True)
    return wb, sectors, dest, capex

# -----------------------------------------------------------------------------
# Charts
# -----------------------------------------------------------------------------
def world_map(df, year, score_col="Score", country_col="Country"):
    d = df[df["Year"] == year].dropna(subset=[score_col, country_col]).copy()
    if d.empty:
        st.info("No data for selected year.")
        return
    fig = px.choropleth(
        d,
        locations=country_col,
        locationmode="country names",
        color=score_col,
        hover_name=country_col,
        color_continuous_scale="Viridis",
        title=f"Country Viability Score — {year}",
    )
    st.plotly_chart(fig, use_container_width=True)

def grade_distribution(df, year):
    d = df[df["Year"] == year]
    if d.empty:
        st.info("No data for selected year.")
        return
    counts = d["Grade"].value_counts().reindex(["A+","A","B","C","D"]).fillna(0)
    fig = go.Figure(
        data=[go.Bar(x=counts.index, y=counts.values)]
    )
    fig.update_layout(title=f"Grade distribution — {year}", xaxis_title="Grade", yaxis_title="# Countries")
    st.plotly_chart(fig, use_container_width=True)

def capex_line(capex_df):
    cols_needed = {"Year", "CAPEX"}
    if not cols_needed.issubset(set(capex_df.columns)):
        st.info("CAPEX sheet needs at least columns: Year, CAPEX.")
        st.dataframe(capex_df.head(10))
        return
    fig = px.line(capex_df.sort_values("Year"), x="Year", y="CAPEX", markers=True, title="Global CAPEX")
    st.plotly_chart(fig, use_container_width=True)

def simple_arima_forecast(series: pd.Series, horizon=5):
    # very small/robust ARIMA: ARIMA(1,1,1) with try/except fallbacks
    s = series.dropna().astype(float)
    s.index = pd.Index(range(len(s)))  # ensure 0..N-1
    if len(s) < 6:
        return None
    try:
        model = ARIMA(s, order=(1,1,1))
        res = model.fit()
        f = res.get_forecast(steps=horizon)
        fc = f.predicted_mean
        lo, hi = f.conf_int().iloc[:,0], f.conf_int().iloc[:,1]
        return fc, lo, hi
    except Exception:
        return None

def capex_forecast_plot(capex_df, horizon=5):
    if not {"Year","CAPEX"}.issubset(capex_df.columns):
        return
    series = capex_df.sort_values("Year")["CAPEX"]
    out = simple_arima_forecast(series, horizon=horizon)
    if out is None:
        st.info("Not enough CAPEX points (or ARIMA failed) to forecast.")
        return
    fc, lo, hi = out
    base = capex_df.sort_values("Year")
    last_year = int(base["Year"].max())
    future_years = [last_year + i for i in range(1, horizon+1)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=base["Year"], y=base["CAPEX"], mode="lines+markers", name="Actual"))
    fig.add_trace(go.Scatter(x=future_years, y=fc, mode="lines+markers", name="Forecast"))
    fig.add_trace(go.Scatter(x=future_years, y=lo, mode="lines", name="Lower", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=future_years, y=hi, mode="lines", name="Upper", line=dict(dash="dash")))
    fig.update_layout(title="CAPEX Forecast (ARIMA)", xaxis_title="Year", yaxis_title="CAPEX")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="FDI Analytics Dashboard", layout="wide")

st.sidebar.title("Data Sources (GitHub)")
st.sidebar.write("Using raw GitHub URLs — no uploads needed.")
st.sidebar.code("\n".join(f"{k}: {raw_url(v)}" for k, v in FILES.items()), language="bash")

with st.sidebar.expander("Settings", expanded=False):
    RAW_BASE = st.text_input("GitHub RAW base URL", RAW_BASE, help="Save & rerun after editing.")
    display_year = st.text_input("Default Year (fallback to latest in data)", "")
    try:
        DEFAULT_YEAR = int(display_year) if display_year.strip() else None
    except Exception:
        DEFAULT_YEAR = None

wb, sectors, dest, capex = load_all()
years = sorted([int(y) for y in wb["Year"].dropna().unique()])
default_year = DEFAULT_YEAR or (years[-1] if years else 2024)

st.title("FDI Analytics Dashboard")
st.caption("EDA • Viability Scoring • Forecasting • Comparisons • Sectors")

# --- Filters
col1, col2, col3 = st.columns([1,1,2])
with col1:
    year_sel = st.selectbox("Year", years, index=max(0, years.index(default_year)) if years else 0)
with col2:
    continents = ["All"] + sorted([c for c in wb.get("Continent", pd.Series(dtype=str)).dropna().unique()])
    cont_sel = st.selectbox("Continent", continents, index=0)
with col3:
    q = st.text_input("Search country (optional)")

# Apply filters
wb_f = wb.copy()
if cont_sel != "All" and "Continent" in wb_f.columns:
    wb_f = wb_f[wb_f["Continent"] == cont_sel]
if q.strip():
    wb_f = wb_f[wb_f["Country"].str.contains(q.strip(), case=False, na=False)]

tabs = st.tabs(["Overview", "EDA", "Scoring", "Forecasting", "Compare", "Sectors"])

# ---------------- Overview ----------------
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Countries tracked", f"{wb['Country'].nunique():,}")
    with c2:
        st.metric("Years", f"{len(years)}")
    with c3:
        med_score = wb[wb["Year"] == year_sel]["Score"].median()
        st.metric("Median Score", f"{med_score:.2f}" if not math.isnan(med_score) else "—")
    with c4:
        top_a = (wb[wb["Year"] == year_sel]["Grade"] == "A").sum() + (wb[wb["Year"] == year_sel]["Grade"] == "A+").sum()
        st.metric("A / A+ Countries", f"{int(top_a)}")

    st.subheader("CAPEX Trend")
    capex_line(capex)

    st.subheader("World Map — Viability Score")
    world_map(wb_f, year_sel)

    c5, c6 = st.columns([2,1])
    with c5:
        st.subheader("Top Countries (sample)")
        if "CAPEX" not in wb_f.columns:
            tmp = wb_f[wb_f["Year"] == year_sel].sort_values("Score", ascending=False)[
                ["Country", "Score", "Grade"]
            ].head(20)
        else:
            tmp = wb_f[wb_f["Year"] == year_sel].sort_values(["Score","CAPEX"], ascending=False)[
                ["Country", "Score", "Grade", "CAPEX"]
            ].head(20)
        st.dataframe(tmp, use_container_width=True)
    with c6:
        st.subheader("Grade Distribution")
        grade_distribution(wb_f, year_sel)

# ---------------- EDA ----------------
with tabs[1]:
    st.subheader("Sector CAPEX breakdown (from merged_sectors_data.csv)")
    if sectors.empty:
        st.info("No sectors data available.")
    else:
        # Expect columns: Year, Sector, CAPEX or share/value
        # trial parsing:
        if "Year" not in sectors.columns:
            for alt in ["year","YEAR"]:
                if alt in sectors.columns:
                    sectors.rename(columns={alt:"Year"}, inplace=True)
        if "Sector" not in sectors.columns:
            for alt in ["sector","SECTOR","Industry"]:
                if alt in sectors.columns:
                    sectors.rename(columns={alt:"Sector"}, inplace=True)
        value_col = None
        for cand in ["CAPEX","Capex","Value","Share","Amount"]:
            if cand in sectors.columns:
                value_col = cand; break
        if value_col:
            s = sectors[sectors["Year"] == year_sel]
            fig = px.pie(s, names="Sector", values=value_col, title=f"Sectors — {year_sel}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(sectors.head(20), use_container_width=True)

    st.divider()
    st.subheader("Destination patterns (from merged_destinations_data.csv)")
    st.dataframe(dest.head(25), use_container_width=True)

# ---------------- Scoring ----------------
with tabs[2]:
    st.subheader("Indicator Weights")
    # show only those that exist in dataframe:
    rows = []
    for k, meta in INDICATOR_WEIGHTS.items():
        w = meta["weight"] if isinstance(meta, dict) else meta
        direction = meta.get("direction", "positive") if isinstance(meta, dict) else "positive"
        rows.append({"Indicator": k, "Weight %": w, "Direction": "+" if direction=="positive" else "-",
                     "Column present": k in wb.columns})
    st.dataframe(pd.DataFrame(rows).sort_values("Weight %", ascending=False), use_container_width=True)

    st.divider()
    st.subheader(f"Country scores — {year_sel}")
    st.dataframe(
        wb_f[wb_f["Year"] == year_sel][["Country","Score","Grade"] + [c for c in wb.columns if c in sum(CATEGORY_BREAKDOWN.values(), [])]].sort_values("Score", ascending=False),
        use_container_width=True,
    )

# ---------------- Forecasting ----------------
with tabs[3]:
    st.subheader("CAPEX Forecast")
    capex_forecast_plot(capex, horizon=5)

# ---------------- Compare ----------------
with tabs[4]:
    st.subheader("Compare two countries")
    countries = sorted(wb_f["Country"].dropna().unique().tolist())
    c1, c2, c3 = st.columns(3)
    with c1:
        a = st.selectbox("Country A", countries, index=0 if countries else None)
    with c2:
        b = st.selectbox("Country B", countries, index=1 if len(countries) > 1 else (0 if countries else None))
    with c3:
        metric = st.selectbox("Metric", ["Score","Grade","GDP growth (annual %)","Inflation, consumer prices (annual %)","GDP per capita, PPP (current international $)"])
    if countries:
        d = wb_f[(wb_f["Country"].isin([a,b])) & (wb_f["Year"]==year_sel)]
        if metric == "Grade":
            st.dataframe(d[["Country","Grade"]], use_container_width=True)
        else:
            st.dataframe(d[["Country", metric, "Score", "Grade"]], use_container_width=True)

# ---------------- Sectors ----------------
with tabs[5]:
    st.subheader("Sector trends (toy line — replace with your preferred view)")
    # We’ll show top N sectors by average value across years if possible
    if not sectors.empty:
        value_col = None
        for cand in ["CAPEX","Capex","Value","Share","Amount"]:
            if cand in sectors.columns:
                value_col = cand; break
        if value_col:
            top = sectors.groupby("Sector")[value_col].mean().sort_values(ascending=False).head(5).index.tolist()
            s = sectors[sectors["Sector"].isin(top)]
            fig = px.line(s, x="Year", y=value_col, color="Sector", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric sector value column found.")
    else:
        st.info("No sectors data loaded.")
