import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

print("===== STAGE 3 EXECUTION STARTED =====")

# ==========================================
# 1️⃣ LOAD TRAIN + TEST (same as Stage 2)
# ==========================================

load_train = pd.read_csv("Electric_Load_Data_Train.csv")
external_train = pd.read_csv("External_Factor_Data_Train.csv")
events = pd.read_csv("Events_Data.csv")

load_train["DATETIME"] = pd.to_datetime(load_train["DATETIME"], format="%d%b%Y:%H:%M:%S", errors='coerce')
external_train["DATETIME"] = pd.to_datetime(external_train["DATETIME"], format="%d%b%Y:%H:%M:%S", errors='coerce')
events["Date"] = pd.to_datetime(events["Date"], errors='coerce')

load_train = load_train.dropna()
external_train = external_train.dropna()
events = events.dropna()

train_data = load_train.merge(external_train, on="DATETIME", how="left")
train_data["DATE"] = train_data["DATETIME"].dt.date
events["DATE"] = events["Date"].dt.date
train_data = train_data.merge(events[["DATE","Holiday_Ind"]], on="DATE", how="left")
train_data["Holiday_Ind"] = train_data["Holiday_Ind"].fillna(0)

train_data["hour"] = train_data["DATETIME"].dt.hour
train_data["day_of_week"] = train_data["DATETIME"].dt.dayofweek
train_data["month"] = train_data["DATETIME"].dt.month
train_data["is_weekend"] = train_data["day_of_week"].isin([5,6]).astype(int)
train_data["is_peak"] = ((train_data["hour"] >= 18) & (train_data["hour"] < 22)).astype(int)

train_data = train_data.sort_values("DATETIME")
train_data["lag_1"] = train_data["LOAD"].shift(1)
train_data["lag_96"] = train_data["LOAD"].shift(96)
train_data["lag_672"] = train_data["LOAD"].shift(672)
train_data["rolling_24h"] = train_data["LOAD"].rolling(96).mean()
train_data["rolling_7d"] = train_data["LOAD"].rolling(672).mean()
train_data = train_data.dropna()

features = [
    "ACT_HEAT_INDEX","ACT_HUMIDITY","ACT_RAIN",
    "ACT_TEMP","COOL_FACTOR","Holiday_Ind",
    "hour","day_of_week","month","is_weekend","is_peak",
    "lag_1","lag_96","lag_672",
    "rolling_24h","rolling_7d"
]

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(train_data[features], train_data["LOAD"])

# ==========================================
# LOAD TEST
# ==========================================

load_test = pd.read_csv("Electric_Load_Data_Test_upd.csv")
external_test = pd.read_csv("External_Factor_Data_Test_upd.csv")

load_test["DATETIME"] = pd.to_datetime(load_test["DATETIME"], format="%d%b%Y:%H:%M:%S", errors='coerce')
external_test["DATETIME"] = pd.to_datetime(external_test["DATETIME"], format="%d%b%Y:%H:%M:%S", errors='coerce')

test = load_test.merge(external_test, on="DATETIME", how="left")
test["DATE"] = test["DATETIME"].dt.date
test = test.merge(events[["DATE","Holiday_Ind"]], on="DATE", how="left")
test["Holiday_Ind"] = test["Holiday_Ind"].fillna(0)

test["hour"] = test["DATETIME"].dt.hour
test["day_of_week"] = test["DATETIME"].dt.dayofweek
test["month"] = test["DATETIME"].dt.month
test["is_weekend"] = test["day_of_week"].isin([5,6]).astype(int)
test["is_peak"] = ((test["hour"] >= 18) & (test["hour"] < 22)).astype(int)

test = test.sort_values("DATETIME")
test["lag_1"] = test["LOAD"].shift(1)
test["lag_96"] = test["LOAD"].shift(96)
test["lag_672"] = test["LOAD"].shift(672)
test["rolling_24h"] = test["LOAD"].rolling(96).mean()
test["rolling_7d"] = test["LOAD"].rolling(672).mean()
test = test.dropna()

test["forecast_base"] = model.predict(test[features])

# ==========================================
# STAGE 3 PENALTY FUNCTION
# ==========================================

def penalty_stage3(actual, forecast, is_peak):
    return np.where(
        (actual > forecast) & (is_peak == 1),
        6 * (actual - forecast),
        np.where(
            (actual > forecast) & (is_peak == 0),
            4 * (actual - forecast),
            2 * (forecast - actual)
        )
    )

# ==========================================
# CONSTRAINT OPTIMIZATION LOOP
# ==========================================

best_solution = None

for m in np.arange(1.000, 1.031, 0.001):  # up to 3% uplift max
    forecast_adj = test["forecast_base"].copy()
    forecast_adj.loc[test["is_peak"]==1] *= m

    penalty = penalty_stage3(test["LOAD"], forecast_adj, test["is_peak"])
    total_penalty = penalty.sum()

    bias = ((forecast_adj - test["LOAD"]).mean() / test["LOAD"].mean()) * 100
    avg_uplift = ((forecast_adj - test["forecast_base"]).mean() / test["forecast_base"].mean()) * 100

    # Peak reliability constraint
    under_5pct = (
        (test["is_peak"]==1) &
        ((test["LOAD"] - forecast_adj)/test["LOAD"] > 0.05)
    ).sum()

    if (
        (-2 <= bias <= 3) and
        (avg_uplift <= 3) and
        (under_5pct <= 3)
    ):
        best_solution = {
            "multiplier": m,
            "total_penalty": total_penalty,
            "bias": bias,
            "avg_uplift": avg_uplift,
            "under_5pct_intervals": under_5pct
        }
        break

if best_solution is None:
    print("⚠ No feasible solution found under constraints.")
else:
    print("\n===== STAGE 3 OPTIMAL SOLUTION =====")
    for k,v in best_solution.items():
        print(f"{k}: {v}")

# ==========================================
# RISK TRANSPARENCY METRICS
# ==========================================

forecast_final = test["forecast_base"].copy()
forecast_final.loc[test["is_peak"]==1] *= best_solution["multiplier"]

abs_dev = abs(test["LOAD"] - forecast_final)
p95 = np.percentile(abs_dev,95)

worst_5 = test.loc[abs_dev.nlargest(5).index,
                    ["DATETIME","LOAD"]].copy()
worst_5["Deviation"] = abs_dev.nlargest(5).values

print("\n95th Percentile Absolute Deviation:", p95)
print("\nWorst 5 Deviation Intervals:")
print(worst_5)

print("\n===== STAGE 3 COMPLETE =====")
