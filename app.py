import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import io
import base64
import tempfile

st.title("🔋 Delhi Electricity Load Forecasting (TA2)")

uploaded_file = st.file_uploader("📂 Upload the CSV (vpgp_data_final.csv)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    data["Timestamp"] = pd.to_datetime(data["Timestamp"])
    st.success("CSV uploaded and loaded successfully.")

    # Feature Engineering
    for col in ["Hour", "Day", "Month", "DayOfWeek"]:
        data[col] = getattr(data["Timestamp"].dt, col.lower())

    for zone in range(1, 6):
        data[f"Zone{zone}_Load_Lag1"] = data[f"Zone{zone}_Load"].shift(1)
        data[f"Zone{zone}_Load_Lag2"] = data[f"Zone{zone}_Load"].shift(2)

    data = data.dropna()

    # Net Load Calculation
    GRID_LOSS_FACTOR = 0.03
    total_gen = data["Solar_Gen"] + data["Wind_Gen"] + data["Thermal_Gen"] + data["Gas_Gen"] + data["Import_Gen"] + data["Battery_Discharge"] - data["Battery_Charge"]
    total_load = sum(data[f"Zone{z}_Load"] for z in range(1, 6))

    for z in range(1, 6):
        zl = data[f"Zone{z}_Load"]
        zg = (zl / total_load) * total_gen
        data[f"Zone{z}_Net_Load"] = zl * (1 + GRID_LOSS_FACTOR) - zg

    features = [
        "Hour", "Day", "Month", "DayOfWeek",
        "Zone1_Load", "Zone2_Load", "Zone3_Load", "Zone4_Load", "Zone5_Load",
        "Solar_Gen", "Wind_Gen", "Thermal_Gen", "Gas_Gen",
        "Battery_Charge", "Battery_Discharge", "Import_Gen",
        "Zone1_Load_Lag1", "Zone2_Load_Lag1", "Zone3_Load_Lag1", "Zone4_Load_Lag1", "Zone5_Load_Lag1",
        "Zone1_Load_Lag2", "Zone2_Load_Lag2", "Zone3_Load_Lag2", "Zone4_Load_Lag2", "Zone5_Load_Lag2"
    ]

    X = data[features]
    models = {}
    metrics = {}

    if st.button("🚀 Run Forecasting Model"):
        with st.spinner("Training models..."):
            for z in range(1, 6):
                for kind in ["Load", "Net_Load"]:
                    target = f"Zone{z}_{kind}"
                    X_train, X_test, y_train, y_test = train_test_split(X, data[target], test_size=0.2, random_state=42)
                    model = GridSearchCV(RandomForestRegressor(random_state=42),
                                         param_grid={"n_estimators": [100], "max_depth": [10], "min_samples_split": [2]},
                                         cv=2, scoring='neg_mean_absolute_error', n_jobs=-1)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    models[target] = model.best_estimator_
                    metrics[target] = {"MAE": round(mae, 2), "RMSE": round(rmse, 2)}

        st.success("✅ All models trained!")

        st.subheader("📊 Evaluation Metrics")
        st.table(pd.DataFrame(metrics).T)

        st.subheader("📈 July 2025 Predictions")
        july = data[data["Timestamp"].dt.month == 7]
        X_july = july[features]
        for z in range(1, 6):
            for kind in ["Load", "Net_Load"]:
                target = f"Zone{z}_{kind}"
                pred = models[target].predict(X_july)

                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(july["Timestamp"], july[target], label="Actual", color="blue")
                ax.plot(july["Timestamp"], pred, label="Predicted", color="red", linestyle="--")
                ax.set_title(f"{target} - July 2025")
                ax.set_xlabel("Date")
                ax.set_ylabel("MW")
                ax.legend()
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)
