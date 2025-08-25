import streamlit as st, plotly.express as px
from lib.data import load_all_local, ensure_columns

st.title("Compare Countries")

df, *_ = load_all_local()
if "scored" in st.session_state:
    df = st.session_state["scored"]

if df is None:
    st.info("Compute scores first or load the panel CSV.")
    st.stop()

df = ensure_columns(df)
country_col = "country" if "country" in df.columns else df.columns[0]
year_col = "year" if "year" in df.columns else "Year"

countries = sorted(df[country_col].unique())
a = st.selectbox("Country A", countries, index=0)
b = st.selectbox("Country B", countries, index=1 if len(countries)>1 else 0)
metric = st.selectbox("Metric", ["composite_score","grade"] + [c for c in df.columns if c.endswith("_norm")])

sub = df[df[country_col].isin([a,b])]
if metric == "grade":
    latest = sub.sort_values(year_col).groupby(country_col).tail(1)
    st.dataframe(latest[[country_col, "grade","composite_score"]], use_container_width=True)
else:
    st.plotly_chart(px.line(sub, x=year_col, y=metric, color=country_col, markers=True, title=f"{metric} over time"),
                    use_container_width=True)
