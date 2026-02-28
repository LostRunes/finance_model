import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter

print("===== GENERATING STAGE 3 GOVERNANCE PROOF =====")

# ==========================================================
# CREATE DIRECTORY STRUCTURE
# ==========================================================

base_dir = "Stage3_Governance_Proof"

folders = {
    "C1": "Constraint_1_Financial_Cap",
    "C2": "Constraint_2_Peak_Reliability",
    "C3": "Constraint_3_Bias_Bound",
    "C4": "Constraint_4_Buffering",
    "C5": "Constraint_5_Risk_Transparency"
}

for f in folders.values():
    os.makedirs(os.path.join(base_dir, f), exist_ok=True)

# ==========================================================
# RE-RUN STAGE 3 BASE MODEL (MULTIPLIER = 1.0)
# ==========================================================

load_train = pd.read_csv("Electric_Load_Data_Train.csv")
external_train = pd.read_csv("External_Factor_Data_Train.csv")
events = pd.read_csv("Events_Data.csv")

load_train["DATETIME"] = pd.to_datetime(load_train["DATETIME"], format="%d%b%Y:%H:%M:%S", errors='coerce')
external_train["DATETIME"] = pd.to_datetime(external_train["DATETIME"], format="%d%b%Y:%H:%M:%S", errors='coerce')
events["Date"] = pd.to_datetime(events["Date"], errors='coerce')

load_train = load_train.dropna()
external_train = external_train.dropna()
events = events.dropna()

train = load_train.merge(external_train, on="DATETIME")
train["DATE"] = train["DATETIME"].dt.date
events["DATE"] = events["Date"].dt.date
train = train.merge(events[["DATE","Holiday_Ind"]], on="DATE", how="left")
train["Holiday_Ind"] = train["Holiday_Ind"].fillna(0)

train["hour"] = train["DATETIME"].dt.hour
train["day_of_week"] = train["DATETIME"].dt.dayofweek
train["month"] = train["DATETIME"].dt.month
train["is_weekend"] = train["day_of_week"].isin([5,6]).astype(int)
train["is_peak"] = ((train["hour"]>=18)&(train["hour"]<22)).astype(int)

train = train.sort_values("DATETIME")
train["lag_1"] = train["LOAD"].shift(1)
train["lag_96"] = train["LOAD"].shift(96)
train["lag_672"] = train["LOAD"].shift(672)
train["rolling_24h"] = train["LOAD"].rolling(96).mean()
train["rolling_7d"] = train["LOAD"].rolling(672).mean()
train = train.dropna()

features = [
"ACT_HEAT_INDEX","ACT_HUMIDITY","ACT_RAIN",
"ACT_TEMP","COOL_FACTOR","Holiday_Ind",
"hour","day_of_week","month","is_weekend","is_peak",
"lag_1","lag_96","lag_672",
"rolling_24h","rolling_7d"
]

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(train[features], train["LOAD"])

# TEST DATA
load_test = pd.read_csv("Electric_Load_Data_Test_upd.csv")
external_test = pd.read_csv("External_Factor_Data_Test_upd.csv")

load_test["DATETIME"] = pd.to_datetime(load_test["DATETIME"], format="%d%b%Y:%H:%M:%S", errors='coerce')
external_test["DATETIME"] = pd.to_datetime(external_test["DATETIME"], format="%d%b%Y:%H:%M:%S", errors='coerce')

test = load_test.merge(external_test, on="DATETIME")
test["DATE"] = test["DATETIME"].dt.date
test = test.merge(events[["DATE","Holiday_Ind"]], on="DATE", how="left")
test["Holiday_Ind"] = test["Holiday_Ind"].fillna(0)

test["hour"] = test["DATETIME"].dt.hour
test["day_of_week"] = test["DATETIME"].dt.dayofweek
test["month"] = test["DATETIME"].dt.month
test["is_weekend"] = test["day_of_week"].isin([5,6]).astype(int)
test["is_peak"] = ((test["hour"]>=18)&(test["hour"]<22)).astype(int)

test = test.sort_values("DATETIME")
test["lag_1"] = test["LOAD"].shift(1)
test["lag_96"] = test["LOAD"].shift(96)
test["lag_672"] = test["LOAD"].shift(672)
test["rolling_24h"] = test["LOAD"].rolling(96).mean()
test["rolling_7d"] = test["LOAD"].rolling(672).mean()
test = test.dropna()

test["forecast"] = model.predict(test[features])

# ==========================================================
# PENALTY FUNCTION
# ==========================================================

def penalty(actual, forecast, is_peak):
    return np.where(
        (actual > forecast) & (is_peak == 1),
        6*(actual-forecast),
        np.where(
            (actual > forecast) & (is_peak == 0),
            4*(actual-forecast),
            2*(forecast-actual)
        )
    )

test["penalty"] = penalty(test["LOAD"], test["forecast"], test["is_peak"])

# ==========================================================
# CONSTRAINT 1 — FINANCIAL
# ==========================================================

total_penalty = test["penalty"].sum()
peak_penalty = test[test["is_peak"]==1]["penalty"].sum()
offpeak_penalty = test[test["is_peak"]==0]["penalty"].sum()

pd.DataFrame([{
"Total_Penalty": total_penalty,
"Peak_Penalty": peak_penalty,
"OffPeak_Penalty": offpeak_penalty
}]).to_csv(os.path.join(base_dir, folders["C1"], "Financial_Summary.csv"), index=False)

# ==========================================================
# CONSTRAINT 2 — PEAK RELIABILITY
# ==========================================================

under_5pct = test[
(test["is_peak"]==1) &
((test["LOAD"]-test["forecast"])/test["LOAD"]>0.05)
]

under_5pct.to_csv(os.path.join(base_dir, folders["C2"], "Peak_Underestimation_Exceeding_5pct.csv"), index=False)

# ==========================================================
# CONSTRAINT 3 — BIAS
# ==========================================================

bias = ((test["forecast"]-test["LOAD"]).mean()/test["LOAD"].mean())*100

pd.DataFrame([{"Forecast_Bias_%": bias}]).to_csv(
os.path.join(base_dir, folders["C3"], "Bias_Summary.csv"), index=False)

# ==========================================================
# CONSTRAINT 4 — BUFFERING
# ==========================================================

avg_uplift = 0.0  # baseline model
pd.DataFrame([{"Average_Uplift_%": avg_uplift}]).to_csv(
os.path.join(base_dir, folders["C4"], "Buffering_Summary.csv"), index=False)

# ==========================================================
# CONSTRAINT 5 — RISK TRANSPARENCY
# ==========================================================

abs_dev = abs(test["LOAD"]-test["forecast"])
p95 = np.percentile(abs_dev,95)

worst5 = test.loc[abs_dev.nlargest(5).index,
["DATETIME","LOAD","forecast"]]
worst5["Deviation"] = abs_dev.nlargest(5).values

pd.DataFrame([{"P95_Absolute_Deviation": p95}]).to_csv(
os.path.join(base_dir, folders["C5"], "P95_Deviation.csv"), index=False)

worst5.to_csv(
os.path.join(base_dir, folders["C5"], "Worst_5_Intervals.csv"), index=False)

# Financial impact of peak volatility
peak_volatility_impact = peak_penalty
pd.DataFrame([{"Peak_Volatility_Financial_Impact": peak_volatility_impact}]).to_csv(
os.path.join(base_dir, folders["C5"], "Peak_Volatility_Impact.csv"), index=False)

# ==========================================================
# GENERATE FINAL PDF REPORT
# ==========================================================

doc = SimpleDocTemplate(os.path.join(base_dir, "Stage3_Governance_Report.pdf"), pagesize=letter)
styles = getSampleStyleSheet()
story = []

story.append(Paragraph("Stage 3 Governance Compliance Report", styles['Title']))
story.append(Spacer(1,12))

story.append(Paragraph("Financial Summary", styles['Heading2']))
story.append(Paragraph(f"Total Penalty: ₹{total_penalty:,.2f}", styles['Normal']))
story.append(Paragraph(f"Peak Penalty: ₹{peak_penalty:,.2f}", styles['Normal']))
story.append(Paragraph(f"Off-Peak Penalty: ₹{offpeak_penalty:,.2f}", styles['Normal']))
story.append(Spacer(1,12))

story.append(Paragraph("Reliability & Bias", styles['Heading2']))
story.append(Paragraph(f"Peak Underestimation >5% Intervals: {len(under_5pct)}", styles['Normal']))
story.append(Paragraph(f"Forecast Bias: {bias:.2f}%", styles['Normal']))
story.append(Spacer(1,12))

story.append(Paragraph("Risk Transparency", styles['Heading2']))
story.append(Paragraph(f"95th Percentile Deviation: {p95:.2f}", styles['Normal']))
story.append(Paragraph("Worst 5 intervals listed separately.", styles['Normal']))

doc.build(story)

print("===== STAGE 3 GOVERNANCE PROOF GENERATED SUCCESSFULLY =====")
print("Check folder: Stage3_Governance_Proof")
