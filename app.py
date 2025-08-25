# lib/data.py
import os, io, json
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler

# Optional: Plotly/Streamlit only where needed to avoid circular imports

DATA_FILES = {
    "scores": "world_bank_data_with_scores_and_continent.csv",
    "sectors": "merged_sectors_data.csv",
    "destinations": "merged_destinations_data.csv",
    "capex_eda": "capex_EDA.xlsx",
}

DEFAULT_WEIGHTS = {
    "GDP growth (annual %)": {"weight": 0.10, "direction": "+"},
    "GDP per capita, PPP (current international $)": {"weight": 0.08, "direction": "+"},
    "Current account balance (% of GDP)": {"weight": 0.06, "direction": "+"},
    "Foreign direct investment, net outflows (% of GDP)": {"weight": 0.06, "direction": "+"},
    "Inflation, consumer prices (annual %)": {"weight": 0.05, "direction": "-"},
    "Exports of goods and services (% of GDP)": {"weight": 0.05, "direction": "+"},
    "Imports of goods and services (% of GDP)": {"weight": 0.05, "direction": "+"},
    "Political Stability and Absence of Violence/Terrorism: Estimate": {"weight": 0.12, "direction": "+"},
    "Government Effectiveness: Estimate": {"weight": 0.10, "direction": "+"},
    "Control of Corruption: Estimate": {"weight": 0.08, "direction": "+"},
    "Access to electricity (% of population)": {"weight": 0.09, "direction": "+"},
    "Individuals using the Internet (% of population)": {"weight": 0.08, "direction": "+"},
    "Total reserves in months of imports": {"weight": 0.08, "direction": "+"},
}

CATEGORY_WEIGHTS = {"Economic": 0.45, "Governance": 0.30, "Infra/Financial": 0.25}

INDICATOR_CATEGORY = {
    "GDP growth (annual %)": "Economic",
    "GDP per capita, PPP (current international $)": "Economic",
    "Current account balance (% of GDP)": "Economic",
    "Foreign direct investment, net outflows (% of GDP)": "Economic",
    "Inflation, consumer prices (annual %)": "Economic",
    "Exports of goods and services (% of GDP)": "Economic",
    "Imports of goods and services (% of GDP)": "Economic",
    "Political Stability and Absence of Violence/Terrorism: Estimate": "Governance",
    "Government Effectiveness: Estimate": "Governance",
    "Control of Corruption: Estimate": "Governance",
    "Access to electricity (% of population)": "Infra/Financial",
    "Individuals using the Internet (% of population)": "Infra/Financial",
    "Total reserves in months of imports": "Infra/Financial",
}

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def load_csv(path: str) -> pd.DataFrame | None:
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def load_excel(path: str, sheet_name=0) -> pd.DataFrame | None:
    if os.path.exists(path):
        return pd.read_excel(path, sheet_name=sheet_name)
    return None

def minmax_by_year(df, cols, year_col="year"):
    out = df.copy()
    for col in cols:
        out[col+"_norm"] = np.nan
    for y, g in out.groupby(year_col):
        sub = g[cols].astype(float)
        for c in cols:
            if sub[c].nunique(dropna=True) <= 1:
                out.loc[g.index, c+"_norm"] = 0.5
            else:
                scaler = MinMaxScaler()
                vals = scaler.fit_transform(sub[[c]].values)
                out.loc[g.index, c+"_norm"] = vals.ravel()
    return out

def apply_direction(df, weights):
    for ind, meta in weights.items():
        col = ind + "_norm"
        if col in df.columns and meta.get("direction") == "-":
            df[col] = 1 - df[col]
    return df

def compute_scores(df, weights, indicator_category, category_weights,
                   country_col="country", year_col="year"):
    df = ensure_columns(df)
    indicators = [k for k in weights.keys() if k in df.columns]
    if not indicators:
        return df, [], "No matching indicator columns found."

    work = df[[country_col, year_col] + indicators].copy()
    work = minmax_by_year(work, indicators, year_col=year_col)
    work = apply_direction(work, weights)

    # category sub-scores
    cats = set(indicator_category.values())
    for cat in cats:
        cat_inds = [i for i in indicators if indicator_category.get(i) == cat]
        cols = [i+"_norm" for i in cat_inds if i+"_norm" in work.columns]
        work[f"{cat}_score"] = work[cols].mean(axis=1) if cols else np.nan

    # composite
    denom = sum(category_weights.values())
    work["composite_score"] = 0
    for cat, w in category_weights.items():
        work["composite_score"] += w * work.get(f"{cat}_score", np.nan)
    work["composite_score"] = work["composite_score"] / denom

    # grades by year
    def grade(p):
        if p >= 0.90: return "A+"
        if p >= 0.75: return "A"
        if p >= 0.50: return "B"
        if p >= 0.25: return "C"
        return "D"
    work["percentile"] = work.groupby(year_col)["composite_score"].rank(pct=True)
    work["grade"] = work["percentile"].apply(grade)
    msg = f"Computed scores for {work[country_col].nunique()} countries across {work[year_col].nunique()} years."
    return work, indicators, msg

def load_all_local():
    scores = load_csv(DATA_FILES["scores"])
    sectors = load_csv(DATA_FILES["sectors"])
    dest = load_csv(DATA_FILES["destinations"])
    capex = load_excel(DATA_FILES["capex_eda"])
    for name, df in [("scores", scores), ("sectors", sectors), ("dest", dest)]:
        if isinstance(df, pd.DataFrame):
            globals()[name] = ensure_columns(df)
    return scores, sectors, dest, capex
