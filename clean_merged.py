import pandas as pd

# =========================
# STEP 1 — LOAD DATA
# =========================
print("Loading datasets...")

load = pd.read_csv("Electric_Load_Data_Train.csv")
external = pd.read_csv("External_Factor_Data_Train.csv")
events = pd.read_csv("Events_Data.csv")

# =========================
# STEP 2 — CLEAN DATETIME
# =========================
print("Cleaning datetime columns...")

load["DATETIME"] = pd.to_datetime(
    load["DATETIME"], 
    format="%d%b%Y:%H:%M:%S", 
    errors="coerce"
)

external["DATETIME"] = pd.to_datetime(
    external["DATETIME"], 
    format="%d%b%Y:%H:%M:%S", 
    errors="coerce"
)

events["Date"] = pd.to_datetime(events["Date"], errors="coerce")

# Drop parsing failures
load = load.dropna(subset=["DATETIME"])
external = external.dropna(subset=["DATETIME"])
events = events.dropna(subset=["Date"])

# =========================
# STEP 3 — SAVE CLEANED BASE FILES
# =========================
print("Saving cleaned individual datasets...")

load.to_csv("Cleaned_Load_Data.csv", index=False)
external.to_csv("Cleaned_External_Data.csv", index=False)
events.to_csv("Cleaned_Events_Data.csv", index=False)

# =========================
# STEP 4 — MERGE LOAD + WEATHER
# =========================
print("Merging load and weather...")

data = load.merge(external, on="DATETIME", how="left")

# =========================
# STEP 5 — MERGE EVENTS
# =========================
data["DATE"] = data["DATETIME"].dt.date
events["DATE"] = events["Date"].dt.date

data = data.merge(
    events[["DATE", "Event_Name", "Holiday_Ind"]],
    on="DATE",
    how="left"
)

data["Holiday_Ind"] = data["Holiday_Ind"].fillna(0)
data["Event_Name"] = data["Event_Name"].fillna("No Event")

# =========================
# STEP 6 — ADD TIME FEATURES (BI FRIENDLY)
# =========================
print("Adding time features...")

data["Year"] = data["DATETIME"].dt.year
data["Month"] = data["DATETIME"].dt.month
data["Month_Name"] = data["DATETIME"].dt.month_name()
data["Day"] = data["DATETIME"].dt.day
data["Hour"] = data["DATETIME"].dt.hour
data["Day_of_Week"] = data["DATETIME"].dt.dayofweek
data["Day_Name"] = data["DATETIME"].dt.day_name()

data["Is_Weekend"] = data["Day_of_Week"].isin([5,6]).astype(int)
data["Is_Peak"] = ((data["Hour"] >= 18) & (data["Hour"] < 22)).astype(int)

# Sort properly
data = data.sort_values("DATETIME")

# =========================
# STEP 7 — SAVE FINAL MERGED DATASET
# =========================
print("Saving merged dataset for PowerBI...")

data.to_csv("Merged_Cleaned_Load_Dataset.csv", index=False)

print("\nDone!")
print("Generated files:")
print(" - Cleaned_Load_Data.csv")
print(" - Cleaned_External_Data.csv")
print(" - Cleaned_Events_Data.csv")
print(" - Merged_Cleaned_Load_Dataset.csv")