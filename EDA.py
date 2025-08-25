import streamlit as st, plotly.express as px
from lib.data import load_all_local, ensure_columns

st.title("EDA — Sectors & Destinations")
_, sectors_df, dest_df, _ = load_all_local()

if sectors_df is not None:
    df = ensure_columns(sectors_df)
    if all(c in df.columns for c in ["year","sector","capex_usd_b"]):
        yr = st.selectbox("Year", sorted(df["year"].unique()))
        topn = st.slider("Top N sectors", 3, 12, 6)
        sub = df[df["year"]==yr].groupby("sector", as_index=False)["capex_usd_b"].sum().sort_values("capex_usd_b", ascending=False).head(topn)
        st.plotly_chart(px.pie(sub, names="sector", values="capex_usd_b", title=f"Sectors — {yr}"), use_container_width=True)
    else:
        st.info("Expected cols in merged_sectors_data.csv: year, sector, capex_usd_b")
else:
    st.warning("Place merged_sectors_data.csv in repo root.")

st.subheader("Raw destinations preview")
st.dataframe(dest_df.head() if dest_df is not None else None, use_container_width=True)
