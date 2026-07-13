import streamlit as st

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Retail Sales Forecasting & Demand Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# Title
# ---------------------------------------------------

st.title("Retail Sales Forecasting & Demand Intelligence Dashboard")

st.markdown("---")

st.write("""
This interactive dashboard presents the complete analysis of the Superstore Sales dataset.

The application combines exploratory data analysis, time series forecasting,
anomaly detection, and product demand segmentation to support inventory planning,
sales analysis, and business decision-making.

Use the navigation panel on the left to explore each dashboard page.
""")

# ---------------------------------------------------
# Project Objective
# ---------------------------------------------------

st.markdown("## Project Objective")

st.write("""
Develop an end-to-end sales forecasting system capable of:

- Understanding historical sales performance
- Predicting future sales demand
- Detecting unusual sales behaviour
- Segmenting products based on demand characteristics
- Supporting inventory planning and business decisions through interactive visualizations
""")

st.markdown("---")

# ---------------------------------------------------
# Dashboard Pages
# ---------------------------------------------------

st.header("Available Dashboard Pages")

col1, col2 = st.columns(2)

with col1:

    st.subheader("1. Sales Overview")

    st.write("""
- Total Sales by Year
- Monthly Sales Trend
- Regional Sales Analysis
- Category-wise Sales Analysis
- Interactive Filters
""")

    st.subheader("2. Forecast Explorer")

    st.write("""
- Category Forecast
- Region Forecast
- Forecast Horizon Selection
- Forecast Visualization
- Model Performance (MAE & RMSE)
""")

with col2:

    st.subheader("3. Anomaly Report")

    st.write("""
- Isolation Forest Detection
- Rolling Z-Score Detection
- Weekly Sales Anomalies
- Detected Anomaly Table
""")

    st.subheader("4. Product Demand Segmentation")

    st.write("""
- K-Means Clustering
- Product Demand Segments
- PCA Cluster Visualization
- Inventory Recommendations
""")

st.markdown("---")

# ---------------------------------------------------
# Technologies Used
# ---------------------------------------------------

st.header("Technologies Used")

c1, c2, c3 = st.columns(3)

with c1:

    st.subheader("Data Analysis")

    st.write("""
- Pandas
- NumPy
- Matplotlib
""")

with c2:

    st.subheader("Machine Learning")

    st.write("""
- XGBoost
- Isolation Forest
- K-Means Clustering
""")

with c3:

    st.subheader("Forecasting")

    st.write("""
- SARIMA
- Prophet
- Time Series Analysis
""")

st.markdown("---")

# ---------------------------------------------------
# About the Dashboard
# ---------------------------------------------------

st.header("About this Dashboard")

st.write("""
This dashboard was developed as part of an internship project to demonstrate the
complete workflow of a retail sales forecasting system.

The project integrates historical sales analysis, forecasting models, anomaly
detection techniques, and demand segmentation into a single interactive
application that can assist business users in making informed inventory and
planning decisions.
""")

st.info(
    "Navigate through the pages using the sidebar to explore different analyses and business insights."
)

st.markdown("---")

st.caption(
    "Developed as part of the Sales Forecasting Internship Project"
)