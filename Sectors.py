import streamlit as st, plotly.express as px
from lib.data import load_all_local, ensure_columns

st.title("Sector Trends")

_, sectors_df, _, _ = load_all_local()
if sectors_df is None:
    st.info("Place merged_sectors_data.csv in repo root.")
    st.stop()

df = ensure_columns(sectors_df)
if all(c in df.columns for c in ["year","sector","capex_usd_b"]):
    s_year = st.selectbox("Year", sorted(df["year"].unique()))
    g = df[df["year"]==s_year].groupby("sector", as_index=False)["capex_usd_b"].sum()
    st.plotly_chart(px.bar(g, x="sector", y="capex_usd_b", title=f"Sector CAPEX — {s_year}"),
                    use_container_width=True)
else:
    st.info("Expected columns: year, sector, capex_usd_b")
