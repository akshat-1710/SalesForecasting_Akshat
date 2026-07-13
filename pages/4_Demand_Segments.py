import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="Product Demand Segmentation",
    layout="wide"
)

# ----------------------------------------------------
# Page Title
# ----------------------------------------------------

st.title("Product Demand Segmentation")

st.write("""
This page groups product sub-categories into demand segments using K-Means
Clustering. The identified demand groups help support inventory planning,
stock allocation, and supply chain decision-making.
""")

st.markdown("---")

# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("train.csv")

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        format="%d/%m/%Y"
    )

    return df


df = load_data()

# ----------------------------------------------------
# Feature Engineering
# ----------------------------------------------------

df["Year"] = df["Order Date"].dt.year

monthly = (
    df.groupby(
        [
            "Sub-Category",
            "Year",
            pd.Grouper(
                key="Order Date",
                freq="ME"
            )
        ]
    )["Sales"]
    .sum()
    .reset_index()
)

feature_table = []

for sub in df["Sub-Category"].unique():

    temp = monthly[
        monthly["Sub-Category"] == sub
    ]

    total_sales = temp["Sales"].sum()

    yearly = temp.groupby("Year")["Sales"].sum()

    growth = yearly.pct_change().mean()

    if pd.isna(growth):
        growth = 0

    volatility = temp["Sales"].std()

    avg_order = df[
        df["Sub-Category"] == sub
    ]["Sales"].mean()

    feature_table.append([
        sub,
        total_sales,
        growth,
        volatility,
        avg_order
    ])

cluster_data = pd.DataFrame(
    feature_table,
    columns=[
        "Sub-Category",
        "Total Sales",
        "Growth Rate",
        "Sales Volatility",
        "Average Order Value"
    ]
)

# ----------------------------------------------------
# Feature Scaling
# ----------------------------------------------------

features = [
    "Total Sales",
    "Growth Rate",
    "Sales Volatility",
    "Average Order Value"
]

scaler = StandardScaler()

scaled = scaler.fit_transform(
    cluster_data[features]
)

# ----------------------------------------------------
# K-Means Clustering
# ----------------------------------------------------

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

cluster_data["Cluster"] = kmeans.fit_predict(scaled)

cluster_names = {

    0: "Premium High Value Products",

    1: "Low Volume Stable Products",

    2: "High Demand Core Products",

    3: "Emerging Growth Products"

}

cluster_data["Demand Segment"] = (
    cluster_data["Cluster"]
    .map(cluster_names)
)

# ----------------------------------------------------
# PCA
# ----------------------------------------------------

pca = PCA(n_components=2)

points = pca.fit_transform(scaled)

cluster_data["PC1"] = points[:,0]

cluster_data["PC2"] = points[:,1]

# ----------------------------------------------------
# Summary Metrics
# ----------------------------------------------------

st.subheader("Demand Segmentation Summary")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Product Sub-Categories",
        len(cluster_data)
    )

with col2:

    st.metric(
        "Demand Segments",
        cluster_data["Demand Segment"].nunique()
    )

st.markdown("---")

# ----------------------------------------------------
# Cluster Visualization
# ----------------------------------------------------

st.subheader("Demand Segment Visualization")

fig, ax = plt.subplots(figsize=(10,7))

colors = [
    "tab:green",
    "tab:orange",
    "tab:blue",
    "tab:red"
]

for cluster in sorted(cluster_data["Cluster"].unique()):

    temp = cluster_data[
        cluster_data["Cluster"] == cluster
    ]

    ax.scatter(
        temp["PC1"],
        temp["PC2"],
        s=120,
        color=colors[cluster],
        label=cluster_names[cluster]
    )

    for _, row in temp.iterrows():

        ax.text(
            row["PC1"] + 0.02,
            row["PC2"] + 0.02,
            row["Sub-Category"],
            fontsize=8
        )

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_title("Product Demand Segmentation using K-Means")

ax.grid(alpha=0.3)

ax.legend()

st.pyplot(fig)

# ----------------------------------------------------
# Segment Details
# ----------------------------------------------------

st.subheader("Demand Segment Details")

display_table = cluster_data[
    [
        "Sub-Category",
        "Demand Segment",
        "Total Sales",
        "Growth Rate",
        "Sales Volatility",
        "Average Order Value"
    ]
].sort_values("Demand Segment")

st.dataframe(
    display_table,
    use_container_width=True
)

# ----------------------------------------------------
# Recommended Inventory Strategy
# ----------------------------------------------------

st.subheader("Recommended Inventory Strategy")

strategy = pd.DataFrame({

"Demand Segment":[

"Premium High Value Products",

"High Demand Core Products",

"Low Volume Stable Products",

"Emerging Growth Products"

],

"Recommended Inventory Strategy":[

"Maintain high inventory levels and prioritize product availability.",

"Maintain frequent replenishment with adequate safety stock.",

"Maintain moderate inventory to minimize holding costs.",

"Increase inventory gradually while monitoring future demand growth."

]

})

st.dataframe(
    strategy,
    use_container_width=True
)

# ----------------------------------------------------
# Business Interpretation
# ----------------------------------------------------

st.subheader("Business Interpretation")

st.markdown("""
The clustering analysis successfully groups product sub-categories into four
distinct demand segments based on historical sales behaviour.

Key business observations include:

- Premium High Value Products generate high revenue and require continuous availability.
- High Demand Core Products contribute significantly to overall sales and should receive inventory priority.
- Low Volume Stable Products exhibit predictable demand and can be managed with moderate inventory levels.
- Emerging Growth Products show increasing demand and should be monitored closely for future inventory expansion.

Using demand segmentation enables more effective inventory planning, optimized
warehouse utilization, and better procurement decisions.
""")

st.markdown("---")

st.success(
    "Demand segmentation supports data-driven inventory planning by enabling different stocking strategies for different product groups."
)