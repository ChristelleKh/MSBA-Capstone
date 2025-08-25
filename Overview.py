import streamlit as st, plotly.express as px, pandas as pd
from lib.data import load_all_local, ensure_columns

st.title("Overview")
scores_df, sectors_df, dest_df, capex = load_all_local()

if scores_df is None:
    st.warning("Place world_bank_data_with_scores_and_continent.csv in the repo root.")
else:
    df = ensure_columns(scores_df)
    country_col = "country" if "country" in df.columns else df.columns[0]
    year_col = "year" if "year" in df.columns else "Year"
    years = sorted(pd.Series(df[year_col]).dropna().unique())
    latest = years[-1] if years else None

    c1, c2, c3 = st.columns(3)
    c1.metric("Countries", f"{df[country_col].nunique():,}")
    c2.metric("Years", f"{len(years)}")
    c3.metric("Latest year", latest if latest is not None else "—")

    if "composite_score" in df.columns and latest is not None:
        d = df[df[year_col] == latest]
        fig = px.choropleth(
            d, locations=country_col, locationmode="country names",
            color="composite_score", hover_name=country_col,
            color_continuous_scale="Viridis", title=f"Composite Score — {latest}"
        )
        fig.update_layout(coloraxis_colorbar_title="Score (0–1)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Compute scores first (see *Scoring* page) to enable the map.")
