import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import os

# Set style for professional look
plt.style.use('ggplot')

print("Generating Report Assets...")

# 1. LOAD DATA
load_train = pd.read_csv("Electric_Load_Data_Train.csv")
external_train = pd.read_csv("External_Factor_Data_Train.csv")
events = pd.read_csv("Events_Data.csv")
load_test = pd.read_csv("Electric_Load_Data_Test_upd.csv")
external_test = pd.read_csv("External_Factor_Data_Test_upd.csv")

# Preprocessing
def preprocess(load, ext, ev):
    load["DATETIME"] = pd.to_datetime(load["DATETIME"], format="%d%b%Y:%H:%M:%S", errors='coerce')
    ext["DATETIME"] = pd.to_datetime(ext["DATETIME"], format="%d%b%Y:%H:%M:%S", errors='coerce')
    ev["Date"] = pd.to_datetime(ev["Date"], errors='coerce')
    
    df = load.merge(ext, on="DATETIME", how="left").dropna()
    df["DATE"] = df["DATETIME"].dt.date
    ev["DATE"] = ev["Date"].dt.date
    df = df.merge(ev[["DATE","Holiday_Ind"]], on="DATE", how="left")
    df["Holiday_Ind"] = df["Holiday_Ind"].fillna(0)
    
    df["hour"] = df["DATETIME"].dt.hour
    df["day_of_week"] = df["DATETIME"].dt.dayofweek
    df["month"] = df["DATETIME"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
    df["is_peak"] = ((df["hour"] >= 18) & (df["hour"] < 22)).astype(int)
    
    df = df.sort_values("DATETIME")
    df["lag_1"] = df["LOAD"].shift(1)
    df["lag_96"] = df["LOAD"].shift(96)
    df["lag_672"] = df["LOAD"].shift(672)
    df["rolling_24h"] = df["LOAD"].rolling(96).mean()
    df["rolling_7d"] = df["LOAD"].rolling(672).mean()
    return df.dropna()

train = preprocess(load_train, external_train, events)
test = preprocess(load_test, external_test, events)

features = [
    "ACT_HEAT_INDEX","ACT_HUMIDITY","ACT_RAIN",
    "ACT_TEMP","COOL_FACTOR","Holiday_Ind",
    "hour","day_of_week","month","is_weekend","is_peak",
    "lag_1","lag_96","lag_672",
    "rolling_24h","rolling_7d"
]

# 2. TRAIN MODEL & FEATURE IMPORTANCE
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(train[features], train["LOAD"])

importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importances.index, importances.values, color='skyblue')
plt.title("Random Forest Feature Importance (Financial Forecast Model)")
plt.xlabel("Importance Score")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("report_assets/feature_importance.png", dpi=300)
plt.close()

# 3. GENERATE STAGE COMPARISONS
test["forecast_base"] = model.predict(test[features])

def calc_penalty(actual, forecast, is_peak, stage):
    # Stage 1: 4x under, 2x over (offpeak/peak same)
    # Stage 2/3: 6x peak under, 4x offpeak under, 2x over
    if stage == 1:
        return np.where(actual > forecast, 4*(actual-forecast), 2*(forecast-actual))
    else:
        return np.where(
            (actual > forecast) & (is_peak == 1), 6*(actual-forecast),
            np.where((actual > forecast) & (is_peak == 0), 4*(actual-forecast), 2*(forecast-actual))
        )

# Stage 1 Baseline (No multiplier)
p1 = calc_penalty(test["LOAD"], test["forecast_base"], test["is_peak"], 1).sum()

# Stage 2 (Optimal multiplier from execution logs was ~1.015)
m2 = 1.015
f2 = test["forecast_base"].copy()
f2.loc[test["is_peak"]==1] *= m2
p2 = calc_penalty(test["LOAD"], f2, test["is_peak"], 2).sum()

# Stage 3 (Governance constrained multiplier ~1.01)
m3 = 1.01
f3 = test["forecast_base"].copy()
f3.loc[test["is_peak"]==1] *= m3
p3 = calc_penalty(test["LOAD"], f3, test["is_peak"], 2).sum()

# 4. PENALTY TRANSITION PLOT
stages = ["Stage 1\n(ML Baseline)", "Stage 2\n(Structural Recal)", "Stage 3\n(Gov-Compliance)"]
penalties = [p1, p2, p3]

plt.figure(figsize=(8, 5))
bars = plt.bar(stages, penalties, color=['#3498db', '#e74c3c', '#27ae60'])
plt.title("Total Financial Penalty across Optimization Stages")
plt.ylabel("Total Penalty (INR)")
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5000, f"₹{yval:,.0f}", ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig("report_assets/penalty_progression.png", dpi=300)
plt.close()

# 5. ACTUAL VS FORECAST (STAGE 3)
plt.figure(figsize=(12, 6))
subset = test.iloc[200:500]
plt.plot(subset["DATETIME"], subset["LOAD"], label="Actual Load", color='black', alpha=0.8)
plt.plot(subset["DATETIME"], f3[subset.index], label="Stage 3 Forecast (Optimized)", color='#27ae60', linestyle='--')
plt.fill_between(subset["DATETIME"], subset["LOAD"], f3[subset.index], alpha=0.1, color='green')
plt.title("Stage 3 Final Optimized Forecast vs Actual Load (Sample Period)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("report_assets/actual_vs_forecast_final.png", dpi=300)
plt.close()

# 6. EXTERNAL CORRELATIONS
corr = train[["LOAD", "ACT_TEMP", "ACT_HUMIDITY", "ACT_RAIN", "ACT_HEAT_INDEX"]].corr()
plt.figure(figsize=(8, 6))
cax = plt.matshow(corr, cmap="RdBu_r")
plt.colorbar(cax)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Load vs Weather Factors Correlation Matrix", pad=20)
plt.tight_layout()
plt.savefig("report_assets/weather_correlation.png", dpi=300)
plt.close()

print("Assets Generated in /report_assets/")
