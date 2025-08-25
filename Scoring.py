import json, streamlit as st, pandas as pd, plotly.express as px
from lib.data import (load_all_local, ensure_columns, DEFAULT_WEIGHTS,
                      INDICATOR_CATEGORY, CATEGORY_WEIGHTS, compute_scores)

st.title("Viability Scoring")

scores_df, *_ = load_all_local()
if scores_df is None:
    st.warning("Load the panel CSV first.")
    st.stop()

df = ensure_columns(scores_df)
country_col = "country" if "country" in df.columns else df.columns[0]
year_col = "year" if "year" in df.columns else "Year"

weights_json = st.text_area("Weights JSON", value=json.dumps(DEFAULT_WEIGHTS, indent=2), height=320)
try:
    weights = json.loads(weights_json)
except Exception as e:
    st.error(f"Invalid JSON: {e}")
    st.stop()

scored, used, msg = compute_scores(df, weights, INDICATOR_CATEGORY, CATEGORY_WEIGHTS, country_col, year_col)
st.caption(msg)
if not used:
    st.error("No indicators matched your columns.")
    st.stop()

st.success(f"Using indicators: {', '.join(used)}")
st.session_state["scored"] = scored

latest_year = int(sorted(scored[year_col].unique())[-1])
latest = scored[scored[year_col]==latest_year]
dist = latest["grade"].value_counts().reindex(["A+","A","B","C","D"]).fillna(0).reset_index()
dist.columns = ["grade","count"]
st.plotly_chart(px.bar(dist, x="grade", y="count", title=f"Grade Distribution — {latest_year}"), use_container_width=True)

st.markdown("*Top countries (latest year)*")
st.dataframe(latest.sort_values("composite_score", ascending=False)[[country_col,"composite_score","grade"]].head(25), use_container_width=True)

st.download_button("Download scored dataset (CSV)", scored.to_csv(index=False).encode("utf-8"),
                   file_name="scored_countries.csv", mime="text/csv")
