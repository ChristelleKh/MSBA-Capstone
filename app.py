import io
import math
import json
import typing as t
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

# -----------------------------------------------------------------------------
# CONFIG – set this to your GitHub repo root that holds the data files
# You can also override it in st.secrets["RAW_BASE"] or from the sidebar.
# Example (repo root): https://raw.githubusercontent.com/ChristelleKh/MSBA-Capstone/main
# -----------------------------------------------------------------------------
RAW_BASE = st.secrets.get(
    "RAW_BASE",
    "https://raw.githubusercontent.com/ChristelleKh/MSBA-Capstone/main",
)

FILES = {
    "world_bank": "world_bank_data_with_scores_and_continent.csv",
    "sectors": "merged_sectors_data.csv",
    "destinations": "merged_destinations_data.csv",
    "capex_eda": "capex_EDA.xlsx",  # Excel sheet
}

# Indicator weights. Negative direction means lower is better.
INDICATOR_WEIGHTS: dict[str, t.Union[int, dict]] = {
    "GDP growth (annual %)": 10,
    "GDP per capita, PPP (current international $)": 8,
    "Current account balance (% of GDP)": 6,
    "Foreign direct investment, net outflows (% of GDP)": 6,
    "Inflation, consumer prices (annual %)": {"weight": 5, "direction": "negative"},
    "Exports of goods and services (% of GDP)": 5,
    "Imports of goods and services (% of GDP)": 5,
    "Political Stability and Absence of Violence/Terrorism: Estimate": 12,
    "Government Effectiveness: Estimate": 10,
    "Control of Corruption: Estimate": 8,
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
# Helpers
# -----------------------------------------------------------------------------
def join_url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}/{path.lstrip('/')}"

def try_fetch(url: str, timeout=60) -> bytes:
    """
    Robust GitHub fetch:
    1) try given URL (usually raw.githubusercontent.com)
    2) if it 404s and looks like githubusercontent, swap to GitHub "blob" -> raw
    3) raise a clear error with the attempted URL
    """
    r = requests.get(url, timeout=timeout)
    if r.ok:
        return r.content
    # Fallback: if someone provided a "blob" base or URL mismatch
    if "github.com" in url and "/blob/" in url:
        alt = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        r2 = requests.get(alt, timeout=timeout)
        if r2.ok:
            return r2.content
    # If base was raw and failed, try a "blob" style conversion the other way round
    if "raw.githubusercontent.com" in url:
        parts = url.replace("https://raw.githubusercontent.com/", "").split("/", 2)
        if len(parts) >= 3:
            user, repo, rest = parts
            alt = f"https://github.com/{user}/{repo}/blob/{rest}"
            r3 = requests.get(alt, timeout=timeout)
            if r3.ok:
                # If blob responds OK, user likely pasted the wrong base — tell them
                raise requests.HTTPError(
                    f"Data URL returned non-raw content. "
                    f"Use RAW URL base like https://raw.githubusercontent.com/<user>/<repo>/<branch> "
                    f"(Tried: {url})"
                )
    # Otherwise fail with context
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        raise requests.HTTPError(f"HTTPError while fetching {url}: {e}") from e
    return r.content  # unreachable

@st.cache_data(show_spinner=False, ttl=600)
def read_csv_from_github(path: str) -> pd.DataFrame:
    url = join_url(RAW_BASE, path)
    content = try_fetch(url)
    # Allow UTF-8 with/without BOM
    return pd.read_csv(io.BytesIO(content), encoding="utf-8")

@st.cache_data(show_spinner=False, ttl=600)
def read_excel_from_github(path: str, sheet_name=0) -> pd.DataFrame:
    url = join_url(RAW_BASE, path)
    content = try_fetch(url)
    # openpyxl required (already in requirements)
    return pd.read_excel(io.BytesIO(content), sheet_name=sheet_name, engine="openpyxl")

def minmax_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mask = s.notna()
    if mask.sum() <= 1:
        return pd.Series(np.nan, index=s.index)
    lo, hi = s[mask].min(), s[mask].max()
    if hi == lo:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)

def _rename_harmonize(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort column harmonization for Year/Country and common alternates."""
    df = df.copy()
    renames = {}
    # Year
    for cand in ["Year", "year", "YEAR", "Year ", "Years", "Fiscal Year"]:
        if cand in df.columns:
            renames[cand] = "Year"
            break
    # Country
    for cand in ["Country", "country", "Country Name", "Country_Name", "CountryName", "Economy"]:
        if cand in df.columns:
            renames[cand] = "Country"
            break
    if renames:
        df = df.rename(columns=renames)
    # Make Year numeric if possible
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    return df

def grade_by_year(df: pd.DataFrame, score_col: str, year_col="Year") -> pd.Series:
    parts = []
    for y, g in df.groupby(year_col, dropna=False):
        p = g[score_col].rank(pct=True)
        ggrade = pd.cut(
            p,
            bins=[0, 0.25, 0.50, 0.75, 0.90, 1.0],
            labels=["D", "C", "B", "A", "A+"],
            include_lowest=True,
        )
        ggrade.index = g.index
        parts.append(ggrade)
    return pd.concat(parts).sort_index()

def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    If Score/Grade already present, keep them.
    Otherwise compute a weighted min-max composite using INDICATOR_WEIGHTS.
    """
    df = _rename_harmonize(df)

    if "Score" in df.columns and "Grade" in df.columns:
        return df

    weight_entries: list[tuple[str, float, pd.Series]] = []
    for key, meta in INDICATOR_WEIGHTS.items():
        w = meta["weight"] if isinstance(meta, dict) else meta
        direction = meta.get("direction", "positive") if isinstance(meta, dict) else "positive"
        if key in df.columns:
            v = minmax_series(df[key])
            if direction == "negative":
                v = 1 - v
            weight_entries.append((key, float(w), v))

    if not weight_entries:
        # Nothing to compute with — return df as-is
        df["Score"] = np.nan
        df["Grade"] = np.nan
        return df

    weights = np.array([w for _, w, _ in weight_entries], dtype=float)
    weights = weights / weights.sum()

    score = np.zeros(len(df), dtype=float)
    for (_, _, norm), w in zip(weight_entries, weights):
        # fill NAs with the mean of the normalized series to avoid dropping rows
        score += w * norm.fillna(norm.mean())

    df["Score"] = score
    # if Year missing, grade globally (still works)
    year_col = "Year" if "Year" in df.columns else None
    if year_col:
        df["Grade"] = grade_by_year(df, "Score", year_col="Year")
    else:
        # single global grading if Year not available
        p = df["Score"].rank(pct=True)
        df["Grade"] = pd.cut(
            p, bins=[0, 0.25, 0.50, 0.75, 0.90, 1.0],
            labels=["D", "C", "B", "A", "A+"], include_lowest=True
        )
    return df

def simple_arima_forecast(series: pd.Series, horizon=5):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 6:
        return None
    s.index = pd.RangeIndex(len(s))  # ensure 0..N-1
    try:
        model = ARIMA(s, order=(1, 1, 1))
        res = model.fit()
    except Exception:
        return None
    try:
        f = res.get_forecast(steps=horizon)
        fc = f.predicted_mean
        ci = f.conf_int()
        lo, hi = ci.iloc[:, 0], ci.iloc[:, 1]
        return fc, lo, hi
    except Exception:
        return None

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=True, ttl=600)
def load_all():
    wb = read_csv_from_github(FILES["world_bank"])
    wb = _rename_harmonize(wb)
    wb = compute_scores(wb)

    sectors = read_csv_from_github(FILES["sectors"])
    sectors = _rename_harmonize(sectors)

    dest = read_csv_from_github(FILES["destinations"])
    dest = _rename_harmonize(dest)

    capex = read_excel_from_github(FILES["capex_eda"])  # first sheet
    capex = _rename_harmonize(capex)
    # Standardize capex value column if possible
    if "CAPEX" not in capex.columns:
        for cand in ["Capex", "capex", "CAPEX ($B)", "Capex ($B)", "Value", "Amount"]:
            if cand in capex.columns:
                capex = capex.rename(columns={cand: "CAPEX"})
                break
    return wb, sectors, dest, capex

# -----------------------------------------------------------------------------
# Charts
# -----------------------------------------------------------------------------
def world_map(df, year, score_col="Score", country_col="Country"):
    if "Year" in df.columns:
        d = df[df["Year"] == year].dropna(subset=[score_col, country_col]).copy()
    else:
        d = df.dropna(subset=[score_col, country_col]).copy()
    if d.empty:
        st.info("No data for the selected year.")
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
    if "Year" not in df.columns:
        st.info("No Year column to show grade distribution per year.")
        return
    d = df[df["Year"] == year]
    if d.empty:
        st.info("No data for selected year.")
        return
    counts = d["Grade"].value_counts().reindex(["A+","A","B","C","D"]).fillna(0)
    fig = go.Figure([go.Bar(x=counts.index, y=counts.values)])
    fig.update_layout(title=f"Grade distribution — {year}", xaxis_title="Grade", yaxis_title="# Countries")
    st.plotly_chart(fig, use_container_width=True)

def capex_line(capex_df):
    need = {"Year", "CAPEX"}
    if not need.issubset(capex_df.columns):
        st.info("CAPEX sheet needs at least columns: Year, CAPEX.")
        st.dataframe(capex_df.head(10))
        return
    base = capex_df.dropna(subset=["Year", "CAPEX"]).sort_values("Year")
    fig = px.line(base, x="Year", y="CAPEX", markers=True, title="Global CAPEX")
    st.plotly_chart(fig, use_container_width=True)

def capex_forecast_plot(capex_df, horizon=5):
    if not {"Year", "CAPEX"}.issubset(capex_df.columns):
        return
    base = capex_df.dropna(subset=["Year", "CAPEX"]).sort_values("Year")
    out = simple_arima_forecast(base["CAPEX"], horizon=horizon)
    if out is None:
        st.info("Not enough CAPEX points (or ARIMA failed) to forecast.")
        return
    fc, lo, hi = out
    last_year = int(base["Year"].max())
    future_years = [last_year + i for i in range(1, horizon + 1)]

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
st.sidebar.code("\n".join(f"{k}: {join_url(RAW_BASE, v)}" for k, v in FILES.items()), language="bash")

with st.sidebar.expander("Settings", expanded=False):
    RAW_BASE = st.text_input(
        "GitHub RAW base URL",
        RAW_BASE,
        help="Example: https://raw.githubusercontent.com/<user>/<repo>/<branch>[/optional-folder]"
    )
    default_year_text = st.text_input("Default Year (optional)", "")
    try:
        DEFAULT_YEAR = int(default_year_text) if default_year_text.strip() else None
    except Exception:
        DEFAULT_YEAR = None

# Load data
wb, sectors, dest, capex = load_all()

# Build year list safely
if "Year" in wb.columns:
    years = sorted(pd.Series(pd.to_numeric(wb["Year"], errors="coerce")).dropna().astype(int).unique())
else:
    years = []

default_year = DEFAULT_YEAR or (years[-1] if years else 2024)

st.title("FDI Analytics Dashboard")
st.caption("EDA • Viability Scoring • Forecasting • Comparisons • Sectors")

# Filters
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    year_idx = years.index(default_year) if years and default_year in years else (len(years)-1 if years else 0)
    year_sel = st.selectbox("Year", years if years else [default_year], index=max(0, year_idx))
with col2:
    continents = ["All"]
    if "Continent" in wb.columns:
        continents += sorted([c for c in wb["Continent"].dropna().unique()])
    cont_sel = st.selectbox("Continent", continents, index=0)
with col3:
    q = st.text_input("Search country (optional)")

# Apply filters
wb_f = wb.copy()
if cont_sel != "All" and "Continent" in wb_f.columns:
    wb_f = wb_f[wb_f["Continent"] == cont_sel]
if q.strip() and "Country" in wb_f.columns:
    wb_f = wb_f[wb_f["Country"].str.contains(q.strip(), case=False, na=False)]

tabs = st.tabs(["Overview", "EDA", "Scoring", "Forecasting", "Compare", "Sectors"])

# ---------------- Overview ----------------
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n_countries = wb["Country"].nunique() if "Country" in wb.columns else len(wb)
        st.metric("Countries tracked", f"{n_countries:,}")
    with c2:
        st.metric("Years", f"{len(years)}")
    with c3:
        med_score = (
            wb[wb["Year"] == year_sel]["Score"].median()
            if "Year" in wb.columns and "Score" in wb.columns
            else (wb["Score"].median() if "Score" in wb.columns else float("nan"))
        )
        st.metric("Median Score", f"{med_score:.2f}" if med_score == med_score else "—")
    with c4:
        if "Year" in wb.columns and "Grade" in wb.columns:
            y = wb[wb["Year"] == year_sel]["Grade"]
            top_a = int((y == "A").sum() + (y == "A+").sum())
        else:
            top_a = 0
        st.metric("A / A+ Countries", f"{top_a}")

    st.subheader("CAPEX Trend")
    capex_line(capex)

    st.subheader("World Map — Viability Score")
    world_map(wb_f, year_sel)

    c5, c6 = st.columns([2, 1])
    with c5:
        st.subheader("Top Countries")
        cols = ["Country", "Score", "Grade"]
        if "CAPEX" in wb_f.columns:
            cols += ["CAPEX"]
        if "Year" in wb_f.columns:
            tmp = wb_f[wb_f["Year"] == year_sel]
        else:
            tmp = wb_f
        if "Score" in tmp.columns:
            tmp = tmp.sort_values("Score", ascending=False)
        st.dataframe(tmp[ [c for c in cols if c in tmp.columns] ].head(20), use_container_width=True)
    with c6:
        st.subheader("Grade Distribution")
        grade_distribution(wb_f, year_sel)

# ---------------- EDA ----------------
with tabs[1]:
    st.subheader("Sector CAPEX breakdown (merged_sectors_data.csv)")
    if len(sectors) == 0:
        st.info("No sectors data available.")
    else:
        # Expect Year, Sector, and a numeric value
        if "Sector" not in sectors.columns:
            for alt in ["sector", "SECTOR", "Industry"]:
                if alt in sectors.columns:
                    sectors = sectors.rename(columns={alt: "Sector"})
                    break
        value_col = None
        for cand in ["CAPEX", "Capex", "Value", "Share", "Amount"]:
            if cand in sectors.columns:
                value_col = cand
                break
        if value_col and "Sector" in sectors.columns:
            s = sectors
            if "Year" in s.columns:
                s = s[s["Year"] == year_sel]
            fig = px.pie(s, names="Sector", values=value_col, title=f"Sectors — {year_sel}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(sectors.head(20), use_container_width=True)

    st.divider()
    st.subheader("Destination patterns (merged_destinations_data.csv)")
    st.dataframe(dest.head(25), use_container_width=True)

# ---------------- Scoring ----------------
with tabs[2]:
    st.subheader("Indicator Weights")
    rows = []
    for k, meta in INDICATOR_WEIGHTS.items():
        w = meta["weight"] if isinstance(meta, dict) else meta
        direction = meta.get("direction", "positive") if isinstance(meta, dict) else "positive"
        rows.append({
            "Indicator": k,
            "Weight %": w,
            "Direction": "+" if direction == "positive" else "-",
            "Column present": k in wb.columns
        })
    st.dataframe(pd.DataFrame(rows).sort_values("Weight %", ascending=False), use_container_width=True)

    st.divider()
    st.subheader(f"Country scores — {year_sel}")
    score_cols = ["Country", "Score", "Grade"]
    # add any present indicators from breakdown
    extra_cols = sum(CATEGORY_BREAKDOWN.values(), [])
    score_cols += [c for c in extra_cols if c in wb.columns]
    if "Year" in wb_f.columns:
        df_scores = wb_f[wb_f["Year"] == year_sel]
    else:
        df_scores = wb_f
    if not df_scores.empty:
        st.dataframe(
            df_scores[[c for c in score_cols if c in df_scores.columns]].sort_values("Score", ascending=False),
            use_container_width=True,
        )
    else:
        st.info("No rows to show for the chosen filters/year.")

# ---------------- Forecasting ----------------
with tabs[3]:
    st.subheader("CAPEX Forecast")
    capex_forecast_plot(capex, horizon=5)

# ---------------- Compare ----------------
with tabs[4]:
    st.subheader("Compare two countries")
    countries = sorted(wb_f["Country"].dropna().unique().tolist()) if "Country" in wb_f.columns else []
    c1, c2, c3 = st.columns(3)
    with c1:
        a = st.selectbox("Country A", countries, index=0 if countries else None)
    with c2:
        b = st.selectbox("Country B", countries, index=1 if len(countries) > 1 else (0 if countries else None))
    with c3:
        metric = st.selectbox(
            "Metric",
            ["Score", "Grade",
             "GDP growth (annual %)",
             "Inflation, consumer prices (annual %)",
             "GDP per capita, PPP (current international $)"]
        )
    if countries:
        if "Year" in wb_f.columns:
            d = wb_f[(wb_f["Country"].isin([a, b])) & (wb_f["Year"] == year_sel)]
        else:
            d = wb_f[wb_f["Country"].isin([a, b])]
        if metric == "Grade":
            st.dataframe(d[["Country", "Grade"]], use_container_width=True)
        else:
            show_cols = [c for c in ["Country", metric, "Score", "Grade"] if c in d.columns]
            st.dataframe(d[show_cols], use_container_width=True)

# ---------------- Sectors ----------------
with tabs[5]:
    st.subheader("Sector trends (top lines)")
    if len(sectors) == 0:
        st.info("No sectors data loaded.")
    else:
        # pick a numeric column
        value_col = None
        for cand in ["CAPEX", "Capex", "Value", "Share", "Amount"]:
            if cand in sectors.columns:
                value_col = cand
                break
        if value_col and "Sector" in sectors.columns:
            top = sectors.groupby("Sector")[value_col].mean().sort_values(ascending=False).head(5).index.tolist()
            s = sectors[sectors["Sector"].isin(top)].copy()
            if "Year" not in s.columns:
                st.info("No Year column in sectors to plot as a line; showing table instead.")
                st.dataframe(s.head(50), use_container_width=True)
            else:
                fig = px.line(s.sort_values("Year"), x="Year", y=value_col, color="Sector", markers=True)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric sector value column found.")
