import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# === RE-RUN STAGE 1 PIPELINE UP TO TEST DATA ===

load = pd.read_csv("Electric_Load_Data_Train.csv")
external = pd.read_csv("External_Factor_Data_Train.csv")
events = pd.read_csv("Events_Data.csv")

load["DATETIME"] = pd.to_datetime(load["DATETIME"], format="%d%b%Y:%H:%M:%S", errors='coerce')
external["DATETIME"] = pd.to_datetime(external["DATETIME"], format="%d%b%Y:%H:%M:%S", errors='coerce')
events["Date"] = pd.to_datetime(events["Date"], errors='coerce')

load = load.dropna()
external = external.dropna()
events = events.dropna()

data = load.merge(external, on="DATETIME", how="left")

data["DATE"] = data["DATETIME"].dt.date
events["DATE"] = events["Date"].dt.date
data = data.merge(events[["DATE","Holiday_Ind"]], on="DATE", how="left")
data["Holiday_Ind"] = data["Holiday_Ind"].fillna(0)

data["hour"] = data["DATETIME"].dt.hour
data["day_of_week"] = data["DATETIME"].dt.dayofweek
data["month"] = data["DATETIME"].dt.month
data["is_weekend"] = data["day_of_week"].isin([5,6]).astype(int)
data["is_peak"] = ((data["hour"] >= 18) & (data["hour"] < 22)).astype(int)

data = data.sort_values("DATETIME")
data["lag_1"] = data["LOAD"].shift(1)
data["lag_96"] = data["LOAD"].shift(96)
data["lag_672"] = data["LOAD"].shift(672)
data["rolling_24h"] = data["LOAD"].rolling(96).mean()
data["rolling_7d"] = data["LOAD"].rolling(672).mean()
data = data.dropna()

train = data[data["DATETIME"] < "2020-01-01"]
test = data[data["DATETIME"] >= "2020-01-01"].copy()

features = [
    "ACT_HEAT_INDEX","ACT_HUMIDITY","ACT_RAIN",
    "ACT_TEMP","COOL_FACTOR","Holiday_Ind",
    "hour","day_of_week","month","is_weekend","is_peak",
    "lag_1","lag_96","lag_672",
    "rolling_24h","rolling_7d"
]

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(train[features], train["LOAD"])

test["forecast"] = model.predict(test[features])
test["naive"] = test["lag_96"]

def penalty(actual, forecast):
    return np.where(actual > forecast,
                    4 * (actual - forecast),
                    2 * (forecast - actual))

test["penalty"] = penalty(test["LOAD"], test["forecast"])
test["penalty_naive"] = penalty(test["LOAD"], test["naive"])

# === EXPORT DETAILED BACKTEST DATA ===

export_cols = [
    "DATETIME","LOAD","forecast","naive",
    "penalty","penalty_naive",
    "is_peak","hour","day_of_week","month","Holiday_Ind"
]

test[export_cols].to_csv("Historical_Backtest_Detail.csv", index=False)

# === EXPORT SUMMARY ===

total_penalty = test["penalty"].sum()
peak_penalty = test[test["is_peak"]==1]["penalty"].sum()
offpeak_penalty = test[test["is_peak"]==0]["penalty"].sum()
bias = ((test["forecast"] - test["LOAD"]).mean() / test["LOAD"].mean()) * 100
p95 = np.percentile(abs(test["LOAD"] - test["forecast"]), 95)
naive_total = test["penalty_naive"].sum()

summary = pd.DataFrame([{
    "Total_Penalty": total_penalty,
    "Peak_Penalty": peak_penalty,
    "OffPeak_Penalty": offpeak_penalty,
    "Bias_%": bias,
    "P95_Abs_Deviation": p95,
    "Naive_Penalty": naive_total,
    "%Reduction_vs_Naive": ((naive_total - total_penalty)/naive_total)*100
}])

summary.to_csv("Historical_Backtest_Summary.csv", index=False)

# === EXPORT MULTIPLIER SWEEP ===

multipliers = [0.995, 1.00, 1.005, 1.01, 1.015]
results = []

for m in multipliers:
    adj_forecast = test["forecast"] * m
    adj_penalty = penalty(test["LOAD"], adj_forecast).sum()
    results.append({"Multiplier": m, "Total_Penalty": adj_penalty})

pd.DataFrame(results).to_csv("Multiplier_Sweep.csv", index=False)

print("PowerBI export files generated successfully.")
