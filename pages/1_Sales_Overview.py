import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Sales Overview",
    layout="wide"
)

# ---------------------------------------------------------
# Page Title
# ---------------------------------------------------------

st.title("Sales Overview")

st.write("""
This dashboard provides a comprehensive overview of historical sales performance.
Use the filters in the sidebar to analyze yearly sales, monthly trends,
regional performance, and category-wise sales distribution.
""")

st.markdown("---")

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("train.csv")

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        format="%d/%m/%Y"
    )

    return df


df = load_data()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.header("Dashboard Filters")

selected_region = st.sidebar.selectbox(
    "Select Region",
    ["All"] + sorted(df["Region"].unique().tolist())
)

selected_category = st.sidebar.selectbox(
    "Select Category",
    ["All"] + sorted(df["Category"].unique().tolist())
)

st.sidebar.info(
    "Changing the filters automatically updates all charts and summary metrics."
)

filtered_df = df.copy()

if selected_region != "All":
    filtered_df = filtered_df[
        filtered_df["Region"] == selected_region
    ]

if selected_category != "All":
    filtered_df = filtered_df[
        filtered_df["Category"] == selected_category
    ]

# ---------------------------------------------------------
# KPI Cards
# ---------------------------------------------------------

st.subheader("Sales Summary")



col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Sales",
        f"${filtered_df['Sales'].sum():,.2f}"
    )

with col2:
    st.metric(
        "Total Orders",
        f"{len(filtered_df):,}"
    )

with col3:
    st.metric(
        "Average Order Value",
        f"${filtered_df['Sales'].mean():,.2f}"
    )



st.markdown("---")

# ---------------------------------------------------------
# Total Sales by Year
# ---------------------------------------------------------

st.subheader("Total Sales by Year")

yearly_sales = (
    filtered_df
    .groupby(filtered_df["Order Date"].dt.year)["Sales"]
    .sum()
)

fig, ax = plt.subplots(figsize=(9,4))

bars = ax.bar(
    yearly_sales.index.astype(str),
    yearly_sales.values
)

ax.set_xlabel("Year")
ax.set_ylabel("Sales")
ax.set_title("Total Sales by Year")

for bar in bars:
    ax.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height(),
        f"{bar.get_height():,.0f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

st.pyplot(fig)

# ---------------------------------------------------------
# Monthly Sales Trend
# ---------------------------------------------------------

st.subheader("Monthly Sales Trend")

monthly_sales = (
    filtered_df
    .groupby(
        pd.Grouper(
            key="Order Date",
            freq="ME"
        )
    )["Sales"]
    .sum()
)

fig, ax = plt.subplots(figsize=(12,4))

ax.plot(
    monthly_sales.index,
    monthly_sales.values,
    linewidth=2
)

ax.grid(alpha=0.3)

ax.set_xlabel("Month")
ax.set_ylabel("Sales")
ax.set_title("Monthly Sales Trend")

st.pyplot(fig)

# ---------------------------------------------------------
# Sales by Region
# ---------------------------------------------------------

st.subheader("Sales by Region")

region_sales = (
    filtered_df
    .groupby("Region")["Sales"]
    .sum()
    .sort_values()
)

fig, ax = plt.subplots(figsize=(8,4))

ax.barh(
    region_sales.index,
    region_sales.values
)

ax.set_xlabel("Sales")
ax.set_title("Sales by Region")

st.pyplot(fig)

# ---------------------------------------------------------
# Sales by Category
# ---------------------------------------------------------

st.subheader("Sales by Category")

category_sales = (
    filtered_df
    .groupby("Category")["Sales"]
    .sum()
)

fig, ax = plt.subplots(figsize=(6,6))

ax.pie(
    category_sales.values,
    labels=category_sales.index,
    autopct="%1.1f%%",
    startangle=90
)

ax.set_title("Sales Distribution by Category")

st.pyplot(fig)

# ---------------------------------------------------------
# Dataset Preview
# ---------------------------------------------------------

st.subheader("Filtered Dataset Preview")

st.dataframe(
    filtered_df.head(20),
    use_container_width=True
)

st.caption(
    f"Showing first 20 of {len(filtered_df):,} filtered records."
)

st.markdown("---")

st.info(
    "All visualizations and summary metrics update automatically based on the selected dashboard filters."
)