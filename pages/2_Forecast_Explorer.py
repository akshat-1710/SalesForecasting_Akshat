import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Forecast Explorer",
    layout="wide"
)

# -------------------------------------------------------
# Page Title
# -------------------------------------------------------

st.title("Forecast Explorer")

st.write("""
Explore future sales forecasts using the best-performing forecasting model (XGBoost).
Forecasts can be generated for individual product categories or regions over a
forecast horizon of one to three months.
""")

st.markdown("---")

# -------------------------------------------------------
# Load Data
# -------------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("train.csv")

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        format="%d/%m/%Y"
    )

    return df


df = load_data()

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

st.sidebar.header("Forecast Settings")

segment_type = st.sidebar.selectbox(
    "Forecast Type",
    ["Category", "Region"]
)

if segment_type == "Category":

    segment = st.sidebar.selectbox(
        "Select Category",
        sorted(df["Category"].unique())
    )

    data = df[df["Category"] == segment]

else:

    segment = st.sidebar.selectbox(
        "Select Region",
        sorted(df["Region"].unique())
    )

    data = df[df["Region"] == segment]

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (Months)",
    1,
    3,
    3
)

st.sidebar.info(
    "Select a category or region and choose the forecast horizon to generate future sales predictions."
)

# -------------------------------------------------------
# Monthly Sales
# -------------------------------------------------------

monthly = (
    data.groupby(
        pd.Grouper(
            key="Order Date",
            freq="ME"
        )
    )["Sales"]
    .sum()
    .reset_index()
)

# -------------------------------------------------------
# Feature Engineering
# -------------------------------------------------------

monthly["Lag_1"] = monthly["Sales"].shift(1)
monthly["Lag_2"] = monthly["Sales"].shift(2)
monthly["Lag_3"] = monthly["Sales"].shift(3)

monthly["Rolling_Mean_3"] = (
    monthly["Sales"]
    .rolling(3)
    .mean()
)

monthly["Month"] = monthly["Order Date"].dt.month
monthly["Quarter"] = monthly["Order Date"].dt.quarter

monthly = monthly.dropna()

features = [
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Rolling_Mean_3",
    "Month",
    "Quarter"
]

X = monthly[features]
y = monthly["Sales"]

train_size = len(monthly) - 3

X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]

y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

# -------------------------------------------------------
# Train Model
# -------------------------------------------------------

model = XGBRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

# -------------------------------------------------------
# Forecast Information
# -------------------------------------------------------

st.subheader("Forecast Configuration")

col1, col2 = st.columns(2)

with col1:
    st.metric("Selected Segment", segment)

with col2:
    st.metric("Forecast Horizon", f"{forecast_horizon} Month(s)")

st.markdown("---")

# -------------------------------------------------------
# Forecast Results
# -------------------------------------------------------

forecast_df = pd.DataFrame({
    "Forecast Month": monthly.iloc[-3:]["Order Date"].dt.strftime("%b-%Y").values,
    "Actual Sales": y_test.values,
    "Forecast Sales": pred
})

forecast_df["Actual Sales"] = forecast_df["Actual Sales"].round(2)
forecast_df["Forecast Sales"] = forecast_df["Forecast Sales"].round(2)

st.subheader("Forecast Results")

st.dataframe(
    forecast_df.head(forecast_horizon),
    use_container_width=True
)

# -------------------------------------------------------
# Forecast Chart
# -------------------------------------------------------

st.subheader("Forecast Visualization")

fig, ax = plt.subplots(figsize=(11,5))

ax.plot(
    monthly["Order Date"],
    monthly["Sales"],
    label="Historical Sales",
    linewidth=2
)

ax.plot(
    monthly["Order Date"].iloc[-3:],
    pred,
    marker="o",
    linewidth=2,
    label="Forecast"
)

ax.grid(alpha=0.3)

ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.set_title(f"Sales Forecast for {segment}")

ax.legend()

st.pyplot(fig)

# -------------------------------------------------------
# Model Performance
# -------------------------------------------------------

mae = abs(y_test - pred).mean()
rmse = ((y_test - pred) ** 2).mean() ** 0.5

st.subheader("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("Mean Absolute Error (MAE)", f"{mae:,.2f}")

with col2:
    st.metric("Root Mean Squared Error (RMSE)", f"{rmse:,.2f}")

st.info(
    "Lower MAE and RMSE values indicate better forecasting accuracy. "
    "These metrics measure how closely the predicted sales match the actual sales."
)

st.markdown("---")

st.success(
    "The generated forecast can support inventory planning, procurement scheduling, "
    "and business decision-making by estimating future sales demand."
)