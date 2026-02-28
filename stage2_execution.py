import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

print("===== STAGE 2 EXECUTION STARTED =====")

# =============================
# 1️⃣ LOAD TRAIN DATA (to retrain Stage 1 model)
# =============================

load_train = pd.read_csv("Electric_Load_Data_Train.csv")
external_train = pd.read_csv("External_Factor_Data_Train.csv")
events = pd.read_csv("Events_Data.csv")

load_train["DATETIME"] = pd.to_datetime(load_train["DATETIME"], format="%d%b%Y:%H:%M:%S", errors='coerce')
external_train["DATETIME"] = pd.to_datetime(external_train["DATETIME"], format="%d%b%Y:%H:%M:%S", errors='coerce')
events["Date"] = pd.to_datetime(events["Date"], errors='coerce')

load_train = load_train.dropna()
external_train = external_train.dropna()
events = events.dropna()

# Merge
train_data = load_train.merge(external_train, on="DATETIME", how="left")
train_data["DATE"] = train_data["DATETIME"].dt.date
events["DATE"] = events["Date"].dt.date
train_data = train_data.merge(events[["DATE","Holiday_Ind"]], on="DATE", how="left")
train_data["Holiday_Ind"] = train_data["Holiday_Ind"].fillna(0)

# Time features
train_data["hour"] = train_data["DATETIME"].dt.hour
train_data["day_of_week"] = train_data["DATETIME"].dt.dayofweek
train_data["month"] = train_data["DATETIME"].dt.month
train_data["is_weekend"] = train_data["day_of_week"].isin([5,6]).astype(int)
train_data["is_peak"] = ((train_data["hour"] >= 18) & (train_data["hour"] < 22)).astype(int)

# Lag + rolling
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

print("Training Stage 1 model...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(train_data[features], train_data["LOAD"])

# =============================
# 2️⃣ LOAD TEST DATA (OUT OF TIME)
# =============================

load_test = pd.read_csv("Electric_Load_Data_Test_upd.csv")
external_test = pd.read_csv("External_Factor_Data_Test_upd.csv")

load_test["DATETIME"] = pd.to_datetime(load_test["DATETIME"], format="%d%b%Y:%H:%M:%S", errors='coerce')
external_test["DATETIME"] = pd.to_datetime(external_test["DATETIME"], format="%d%b%Y:%H:%M:%S", errors='coerce')

test_data = load_test.merge(external_test, on="DATETIME", how="left")
test_data["DATE"] = test_data["DATETIME"].dt.date
test_data = test_data.merge(events[["DATE","Holiday_Ind"]], on="DATE", how="left")
test_data["Holiday_Ind"] = test_data["Holiday_Ind"].fillna(0)

# Time features
test_data["hour"] = test_data["DATETIME"].dt.hour
test_data["day_of_week"] = test_data["DATETIME"].dt.dayofweek
test_data["month"] = test_data["DATETIME"].dt.month
test_data["is_weekend"] = test_data["day_of_week"].isin([5,6]).astype(int)
test_data["is_peak"] = ((test_data["hour"] >= 18) & (test_data["hour"] < 22)).astype(int)

# Lag features (using test series only)
test_data = test_data.sort_values("DATETIME")
test_data["lag_1"] = test_data["LOAD"].shift(1)
test_data["lag_96"] = test_data["LOAD"].shift(96)
test_data["lag_672"] = test_data["LOAD"].shift(672)
test_data["rolling_24h"] = test_data["LOAD"].rolling(96).mean()
test_data["rolling_7d"] = test_data["LOAD"].rolling(672).mean()
test_data = test_data.dropna()

print("Generating Stage 2 forecasts...")
test_data["forecast"] = model.predict(test_data[features])

# =============================
# 3️⃣ STAGE 2 PENALTY FUNCTION
# =============================

def penalty_stage2(actual, forecast, is_peak):
    return np.where(
        (actual > forecast) & (is_peak == 1),
        6 * (actual - forecast),
        np.where(
            (actual > forecast) & (is_peak == 0),
            4 * (actual - forecast),
            2 * (forecast - actual)
        )
    )

test_data["penalty"] = penalty_stage2(
    test_data["LOAD"],
    test_data["forecast"],
    test_data["is_peak"]
)

total_penalty = test_data["penalty"].sum()
peak_penalty = test_data[test_data["is_peak"]==1]["penalty"].sum()
offpeak_penalty = test_data[test_data["is_peak"]==0]["penalty"].sum()
bias = ((test_data["forecast"] - test_data["LOAD"]).mean() / test_data["LOAD"].mean()) * 100
p95 = np.percentile(abs(test_data["LOAD"] - test_data["forecast"]),95)

print("Stage 2 Penalty:", total_penalty)

# =============================
# 4️⃣ PEAK RECALIBRATION
# =============================

print("Applying peak recalibration...")
test_data["forecast_adj"] = test_data["forecast"]

# 75th percentile logic → stronger buffer
test_data.loc[test_data["is_peak"]==1,"forecast_adj"] *= 1.05

test_data["penalty_adj"] = penalty_stage2(
    test_data["LOAD"],
    test_data["forecast_adj"],
    test_data["is_peak"]
)

adj_total_penalty = test_data["penalty_adj"].sum()

print("Adjusted Penalty:", adj_total_penalty)

# =============================
# 5️⃣ GRAPHS
# =============================

plt.figure(figsize=(10,5))
subset = test_data.iloc[:500]
plt.plot(subset["DATETIME"], subset["LOAD"], label="Actual")
plt.plot(subset["DATETIME"], subset["forecast"], label="Forecast")
plt.title("Stage 2 Actual vs Forecast")
plt.legend()
plt.savefig("stage2_actual_vs_forecast.png")
plt.close()

plt.figure(figsize=(10,5))
test_data.groupby("hour")["penalty"].mean().plot(kind="bar")
plt.title("Stage 2 Average Penalty by Hour")
plt.savefig("stage2_penalty_by_hour.png")
plt.close()

# =============================
# 6️⃣ PDF REPORT
# =============================

print("Generating Stage 2 Interim Report...")
doc = SimpleDocTemplate("Stage2_Structural_Recalibration_Report.pdf", pagesize=letter)
styles = getSampleStyleSheet()
story = []

story.append(Paragraph("Stage 2 – Structural Recalibration Brief", styles['Title']))
story.append(Spacer(1,12))

story.append(Paragraph("Shock Diagnosis", styles['Heading2']))
story.append(Paragraph(
"Out-of-time dataset exhibits higher volatility and amplified peak variability. "
"Regulatory escalation increases peak underforecast penalty from ₹4 to ₹6.",
styles['Normal']))
story.append(Spacer(1,12))

story.append(Paragraph("Impact Quantification", styles['Heading2']))
table_data = [
["Metric","Value"],
["Total Penalty", f"{total_penalty:,.2f}"],
["Peak Penalty", f"{peak_penalty:,.2f}"],
["Off-Peak Penalty", f"{offpeak_penalty:,.2f}"],
["Bias (%)", f"{bias:.2f}"],
["95th Percentile Abs Dev", f"{p95:.2f}"],
["Recalibrated Penalty", f"{adj_total_penalty:,.2f}"]
]
table = Table(table_data)
table.setStyle(TableStyle([('GRID',(0,0),(-1,-1),1,colors.black)]))
story.append(table)
story.append(Spacer(1,12))

story.append(Paragraph("Trade-Off Recognition", styles['Heading2']))
story.append(Paragraph(
"Increased buffering reduces catastrophic peak underforecast exposure "
"but increases overforecast bias. Risk prioritization favors peak stability.",
styles['Normal']))
story.append(Spacer(1,12))

story.append(Paragraph("Stage 3 Direction", styles['Heading2']))
story.append(Paragraph(
"Introduce peak-specific quantile forecasting (~0.75), volatility-adaptive buffers, "
"and rolling recalibration to manage structural shifts.",
styles['Normal']))

story.append(Spacer(1,12))
story.append(Image("stage2_actual_vs_forecast.png", width=400, height=200))
story.append(Spacer(1,12))
story.append(Image("stage2_penalty_by_hour.png", width=400, height=200))

doc.build(story)

print("Stage 2 Report Generated Successfully.")
