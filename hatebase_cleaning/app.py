import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Hatebase Analysis Dashboard", layout="wide")
st.title("Hatebase Dataset Analysis")

# -------------------------
# File upload section
# -------------------------
st.sidebar.header("Upload datasets")

before_file = st.sidebar.file_uploader("Upload ORIGINAL dataset", type=["parquet"])
after_file = st.sidebar.file_uploader("Upload CLEANED dataset", type=["parquet"])

if before_file is not None and after_file is not None:
    before = pd.read_parquet(before_file)
    after = pd.read_parquet(after_file)

    # -------------------------
    # Overview
    # -------------------------
    st.header("Dataset Overview")

    col1, col2 = st.columns(2)
    col1.markdown("### Rows BEFORE cleaning")
    col1.metric(label="", value=f"{before.shape[0]:,}")
    col2.markdown("### Rows AFTER cleaning")
    col2.metric(label="", value=f"{after.shape[0]:,}")

    st.write("Columns:", after.columns.tolist())

    # -------------------------
    # Class distribution comparison (Histogram)
    # -------------------------
    st.header("Class Distribution: Before vs After (Histogram)")

    before_copy = before.copy()
    before_copy["version"] = "Before"
    after_copy = after.copy()
    after_copy["version"] = "After"
    combined = pd.concat([before_copy, after_copy], ignore_index=True)

    fig_class = px.histogram(
        combined,
        x="class",
        color="version",
        barmode="group",
        title="Class Distribution Before vs After",
        color_discrete_map={"Before": "#1f77b4", "After": "#ff7f0e"},
    )
    fig_class.update_layout(template="simple_white", xaxis_title="Class", yaxis_title="Count")
    st.plotly_chart(fig_class, use_container_width=True)

    # -------------------------
    # Safe vs Unsafe overall (After): Pie chart + counts
    # -------------------------
    st.header("Safe vs Unsafe (After Cleaning)")

    overall_counts = after["class"].value_counts().reset_index()
    overall_counts.columns = ["class", "count"]
    overall_counts["percent"] = (overall_counts["count"] / overall_counts["count"].sum() * 100).round(1)

    colA, colB = st.columns([1, 1])

    with colA:
        st.subheader("Counts & Percentages")
        st.dataframe(overall_counts, use_container_width=True)

    with colB:
        st.subheader("Pie Chart")
        fig_pie = px.pie(
            overall_counts,
            names="class",
            values="count",
            title="Overall Safe vs Unsafe",
            color="class",
            color_discrete_map={"safe": "#2ecc71", "unsafe": "#e74c3c"},
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(template="simple_white")
        st.plotly_chart(fig_pie, use_container_width=True)

    # -------------------------
    # Safe vs Unsafe by Data Source: counts + percentages + chart
    # -------------------------
    st.header("Safe vs Unsafe by Data Source")

    # Counts table
    source_counts = pd.crosstab(after["data"], after["class"])
    source_counts["Total"] = source_counts.sum(axis=1)

    # Percent table (row-normalized)
    source_percent = pd.crosstab(after["data"], after["class"], normalize="index") * 100
    source_percent = source_percent.round(1)
    source_percent["Total"] = 100.0

    # Combined table: safe_count, unsafe_count, safe_pct, unsafe_pct, total
    combined_table = source_counts.join(
        source_percent.add_suffix("_pct"),
        how="left"
    )

    # Put totals first/last nicely (optional)
    cols_order = []
    for c in ["safe", "unsafe", "Total", "safe_pct", "unsafe_pct", "Total_pct"]:
        if c in combined_table.columns:
            cols_order.append(c)
    combined_table = combined_table[cols_order]

    st.subheader("Table (Counts + Percentages)")
    st.dataframe(combined_table, use_container_width=True)

    # Stacked bar chart (counts)
    st.subheader("Visualization (Counts)")
    src_long = source_counts.reset_index().melt(id_vars="data", var_name="class", value_name="count")
    src_long = src_long[src_long["class"].isin(["safe", "unsafe"])]

    fig_src = px.bar(
        src_long,
        x="data",
        y="count",
        color="class",
        title="Safe vs Unsafe by Data Source (Counts)",
        color_discrete_map={"safe": "#2ecc71", "unsafe": "#e74c3c"},
    )
    fig_src.update_layout(
        template="simple_white",
        xaxis_title="Source dataset",
        yaxis_title="Count",
        xaxis_tickangle=-45,
        legend_title_text="Class",
    )
    st.plotly_chart(fig_src, use_container_width=True)

    # Pie chart by source (choose a source)
    st.subheader("Pie Chart by Source")
    selected_source = st.selectbox("Select a source dataset", sorted(after["data"].unique()))

    one = after[after["data"] == selected_source]["class"].value_counts().reset_index()
    one.columns = ["class", "count"]

    fig_pie_src = px.pie(
        one,
        names="class",
        values="count",
        title=f"Safe vs Unsafe for {selected_source}",
        color="class",
        color_discrete_map={"safe": "#2ecc71", "unsafe": "#e74c3c"},
    )
    fig_pie_src.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie_src.update_layout(template="simple_white")
    st.plotly_chart(fig_pie_src, use_container_width=True)

    # -------------------------
    # Browse tweets
    # -------------------------
    st.header("Browse Tweets")

    selected_class = st.selectbox("Select class", sorted(after["class"].unique()))
    subset = after[after["class"] == selected_class]

    st.write(f"Showing random examples from: **{selected_class}** (n={len(subset):,})")
    sample_n = st.slider("How many examples?", min_value=5, max_value=50, value=20, step=5)
    st.dataframe(subset[["tweet", "data"]].sample(min(sample_n, len(subset))), use_container_width=True)

    # -------------------------
    # Search tool
    # -------------------------
    st.header("Search Tweets")

    query = st.text_input("Enter keyword")
    if query:
        results = after[after["tweet"].astype(str).str.contains(query, case=False, na=False)]
        st.write(f"**{len(results):,}** results found")
        st.dataframe(results[["tweet", "class", "data"]].head(100), use_container_width=True)

else:
    st.info("Upload both original and cleaned datasets to begin.")
