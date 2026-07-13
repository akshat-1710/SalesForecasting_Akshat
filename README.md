# Sales Forecasting and Demand Analysis Dashboard

## Project Overview

This project analyzes the Superstore Sales dataset to understand historical sales trends, forecast future sales, detect anomalies, and segment products based on demand characteristics.

The project combines statistical forecasting, machine learning, clustering, anomaly detection, and interactive visualization using Streamlit.

---

## Objectives

- Analyze historical sales performance
- Forecast future sales using multiple forecasting models
- Detect unusual sales behaviour
- Segment products based on demand
- Build an interactive dashboard for business users

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Statsmodels
- Prophet
- XGBoost
- Streamlit

---

## Forecasting Models

- SARIMA
- Facebook Prophet
- XGBoost Regressor

XGBoost achieved the best forecasting performance and was selected for deployment.

---

## Dashboard Pages

### Page 1 – Sales Overview

- Total yearly sales
- Monthly sales trend
- Sales by region
- Sales by category

### Page 2 – Forecast Explorer

- Forecast by category or region
- 1–3 month forecasting
- MAE and RMSE evaluation

### Page 3 – Anomaly Report

- Weekly anomaly detection
- Isolation Forest
- Business interpretation

### Page 4 – Product Demand Segmentation

- K-Means clustering
- PCA visualization
- Demand segments
- Stocking recommendations

---

## Folder Structure

```
SalesForecasting_Akshat/

analysis.ipynb
app.py
requirements.txt
README.md
train.csv

pages/
    1_Sales_Overview.py
    2_Forecast_Explorer.py
    3_Anomaly_Report.py
    4_Demand_Segments.py
```

---

## Run Locally

Install dependencies

```
pip install -r requirements.txt
```

Run the application

```
streamlit run app.py
```

---

## Live Demo

### Streamlit Application

https://salesforecastingakshat-bnjoq4dqhr6kdqbkmhitj8.streamlit.app

### GitHub Repository

https://github.com/akshat-1710/SalesForecasting_Akshat

---

## Author

Akshat