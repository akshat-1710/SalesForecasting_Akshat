import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="Anomaly Report",
    layout="wide"
)

# ----------------------------------------------------
# Page Title
# ----------------------------------------------------

st.title("Sales Anomaly Report")

st.write("""
This page identifies unusual weekly sales patterns using the Isolation Forest
algorithm. Detected anomalies may represent unusually high or low sales weeks
that require further business investigation.
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
# Weekly Sales Aggregation
# ----------------------------------------------------

weekly_sales = (
    df.groupby(
        pd.Grouper(
            key="Order Date",
            freq="W"
        )
    )["Sales"]
    .sum()
    .reset_index()
)

# ----------------------------------------------------
# Isolation Forest
# ----------------------------------------------------

model = IsolationForest(
    contamination=0.05,
    random_state=42
)

weekly_sales["Prediction"] = model.fit_predict(
    weekly_sales[["Sales"]]
)

weekly_sales["Anomaly"] = (
    weekly_sales["Prediction"] == -1
)

anomalies = weekly_sales[
    weekly_sales["Anomaly"]
]

# ----------------------------------------------------
# KPI Cards
# ----------------------------------------------------

st.subheader("Anomaly Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Weeks",
        len(weekly_sales)
    )

with col2:
    st.metric(
        "Detected Anomalies",
        len(anomalies)
    )

with col3:
    anomaly_rate = (len(anomalies) / len(weekly_sales)) * 100
    st.metric(
        "Anomaly Rate",
        f"{anomaly_rate:.1f}%"
    )

st.markdown("---")

# ----------------------------------------------------
# Time Series Chart
# ----------------------------------------------------

st.subheader("Weekly Sales with Detected Anomalies")

fig, ax = plt.subplots(figsize=(14,6))

ax.plot(
    weekly_sales["Order Date"],
    weekly_sales["Sales"],
    label="Weekly Sales",
    linewidth=2
)

ax.scatter(
    anomalies["Order Date"],
    anomalies["Sales"],
    color="red",
    s=70,
    label="Detected Anomaly"
)

ax.set_xlabel("Week")
ax.set_ylabel("Sales")
ax.set_title("Weekly Sales Anomaly Detection using Isolation Forest")

ax.legend()

ax.grid(alpha=0.3)

st.pyplot(fig)

# ----------------------------------------------------
# Detected Anomalies Table
# ----------------------------------------------------

st.subheader("Detected Anomaly Details")

display_table = anomalies[
    [
        "Order Date",
        "Sales"
    ]
].copy()

display_table["Order Date"] = (
    display_table["Order Date"]
    .dt.strftime("%d-%b-%Y")
)

display_table["Sales"] = (
    display_table["Sales"]
    .round(2)
)

st.dataframe(
    display_table,
    use_container_width=True
)

# ----------------------------------------------------
# Business Interpretation
# ----------------------------------------------------

st.subheader("Business Interpretation")

st.markdown("""
The detected anomalies represent weeks where sales deviated significantly from
the normal weekly sales pattern.

Possible business explanations include:

- Festive shopping periods resulting in unusually high sales.
- Promotional campaigns, seasonal discounts, or marketing events.
- Large corporate or bulk customer purchases.
- Inventory shortages or supply chain disruptions.
- Temporary reductions in customer demand during off-peak periods.

These anomalies should be reviewed by the sales and supply chain teams to
determine the underlying business causes and improve future demand planning.
""")

st.markdown("---")

# ----------------------------------------------------
# Recommendation
# ----------------------------------------------------

st.subheader("Recommendation")

st.info(
    "Regular anomaly monitoring can help identify unexpected demand changes, "
    "evaluate promotional performance, detect operational issues, and improve "
    "forecasting accuracy and inventory planning."
)

st.markdown("---")

st.success(
    "The detected anomalies provide valuable insights for proactive business "
    "decision-making and supply chain management."
)