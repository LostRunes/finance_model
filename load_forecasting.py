import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import sys

# Step 1: Load Data
print("Loading datasets...")
load = pd.read_csv("Electric_Load_Data_Train.csv")
external = pd.read_csv("External_Factor_Data_Train.csv")
events = pd.read_csv("Events_Data.csv")

# Convert datetime with error handling
print("Converting datetimes...")
# The format %d%b%Y:%H:%M:%S seems correct for '01APR2013:00:15:00'
load["DATETIME"] = pd.to_datetime(load["DATETIME"], format="%d%b%Y:%H:%M:%S", errors='coerce')
external["DATETIME"] = pd.to_datetime(external["DATETIME"], format="%d%b%Y:%H:%M:%S", errors='coerce')

# Check for parsing failures
if load["DATETIME"].isna().any():
    print(f"Warning: {load['DATETIME'].isna().sum()} rows in load data failed datetime parsing.")
if external["DATETIME"].isna().any():
    print(f"Warning: {external['DATETIME'].isna().sum()} rows in external data failed datetime parsing.")

# Drop rows with failed datetime parsing
load = load.dropna(subset=["DATETIME"])
external = external.dropna(subset=["DATETIME"])

# Events date conversion
events["Date"] = pd.to_datetime(events["Date"], errors='coerce')
events = events.dropna(subset=["Date"])

# 🔥 STEP 2 — Merge Load + Weather
print("Merging Load and Weather data...")
data = load.merge(external, on="DATETIME", how="left")

# 🔥 STEP 3 — Merge Events
data["DATE"] = data["DATETIME"].dt.date
events["DATE"] = events["Date"].dt.date

data = data.merge(events[["DATE", "Holiday_Ind"]], on="DATE", how="left")
data["Holiday_Ind"] = data["Holiday_Ind"].fillna(0)

# 🔥 STEP 4 — Feature Engineering
print("Feature engineering...")
data["hour"] = data["DATETIME"].dt.hour
data["day_of_week"] = data["DATETIME"].dt.dayofweek
data["month"] = data["DATETIME"].dt.month
data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)
data["is_peak"] = ((data["hour"] >= 18) & (data["hour"] < 22)).astype(int)

# 🔥 STEP 5 — Lag Features
# 96 intervals = 1 day, 672 intervals = 7 days
print("Creating lag and rolling features...")
data = data.sort_values("DATETIME")
data["lag_1"] = data["LOAD"].shift(1)
data["lag_96"] = data["LOAD"].shift(96)
data["lag_672"] = data["LOAD"].shift(672)

data["rolling_24h"] = data["LOAD"].rolling(96).mean()
data["rolling_7d"] = data["LOAD"].rolling(672).mean()

# Drop NA rows
data = data.dropna()

# 🔥 STEP 6 — Train-Test Split (TIME BASED)
print("Splitting data...")
train = data[data["DATETIME"] < "2020-01-01"]
test = data[data["DATETIME"] >= "2020-01-01"].copy()

if len(train) == 0 or len(test) == 0:
    print("Error: Train or Test set is empty. Check datetime ranges.")
    sys.exit(1)

# 🔥 STEP 7 — Model
features = [
    "ACT_HEAT_INDEX", "ACT_HUMIDITY", "ACT_RAIN",
    "ACT_TEMP", "COOL_FACTOR", "Holiday_Ind",
    "hour", "day_of_week", "month", "is_weekend", "is_peak",
    "lag_1", "lag_96", "lag_672",
    "rolling_24h", "rolling_7d"
]

print(f"Training RandomForestRegressor on {len(train)} rows...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # Use n_jobs=-1 for speed
model.fit(train[features], train["LOAD"])

print("Generating forecasts...")
test.loc[:, "forecast"] = model.predict(test[features])

# 🔥 STEP 8 — Financial Penalty Function
def penalty(actual, forecast):
    return np.where(actual > forecast,
                    4 * (actual - forecast),
                    2 * (forecast - actual))

test.loc[:, "penalty"] = penalty(test["LOAD"], test["forecast"])

# 🔥 STEP 9 — Compute Required Metrics
total_penalty = test["penalty"].sum()
peak_penalty = test[test["is_peak"] == 1]["penalty"].sum()
offpeak_penalty = test[test["is_peak"] == 0]["penalty"].sum()
bias = ((test["forecast"] - test["LOAD"]).mean() / test["LOAD"].mean()) * 100
p95_abs_dev = np.percentile(abs(test["LOAD"] - test["forecast"]), 95)

print("\n--- Initial Metrics ---")
print(f"Total Penalty: {total_penalty:,.2f}")
print(f"Peak Penalty: {peak_penalty:,.2f}")
print(f"Off-Peak Penalty: {offpeak_penalty:,.2f}")
print(f"Bias: {bias:.2f}%")
print(f"P95 Absolute Deviation: {p95_abs_dev:.2f}")

# 🔥 STEP 10 — Run the Multiplier Sweep
print("\n--- Multiplier Sweep ---")
multipliers = [0.995, 1.00, 1.005, 1.01, 1.015]
results = {}

for m in multipliers:
    adj_forecast = test["forecast"] * m
    adj_penalty = penalty(test["LOAD"], adj_forecast).sum()
    results[m] = adj_penalty
    print(f"Multiplier {m}: {adj_penalty:,.2f}")

best_m = min(results, key=results.get)
print(f"\nBest Multiplier: {best_m} with penalty {results[best_m]:,.2f}")

# 🔥 STEP 11 — Add Naive Baseline
print("\n--- Naive Baseline Comparison ---")
test.loc[:, "naive"] = test["lag_96"]
test.loc[:, "penalty_naive"] = penalty(test["LOAD"], test["naive"])
naive_total_penalty = test["penalty_naive"].sum()

print(f"Naive Penalty: {naive_total_penalty:,.2f}")
print(f"Model Penalty: {total_penalty:,.2f}")

if naive_total_penalty > total_penalty:
    print("WIN: Model outperforms Naive baseline!")
else:
    print("WARNING: Naive baseline is better than or equal to Model.")
