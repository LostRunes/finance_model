import pandas as pd

# Load your Stage 2/3 processed file
data = pd.read_csv("Electric_Load_Data_Test_upd.csv")
external = pd.read_csv("External_Factor_Data_Test_upd.csv")

data["DATETIME"] = pd.to_datetime(data["DATETIME"], format="%d%b%Y:%H:%M:%S")
external["DATETIME"] = pd.to_datetime(external["DATETIME"], format="%d%b%Y:%H:%M:%S")

df = data.merge(external, on="DATETIME", how="left")

# Recreate features (same as model)
df["hour"] = df["DATETIME"].dt.hour
df["is_peak"] = ((df["hour"] >= 18) & (df["hour"] < 22)).astype(int)

# Load predictions from Stage 3 (IMPORTANT)
stage3 = pd.read_csv("Stage3_Final_Output.csv")  # create this if not yet
stage3["DATETIME"] = pd.to_datetime(stage3["DATETIME"])

df = df.merge(stage3[["DATETIME","forecast","forecast_adj","penalty","penalty_adj"]],
              on="DATETIME", how="left")

df = df[df["forecast"].notna()]

# Deviation
df["deviation"] = abs(df["LOAD"] - df["forecast"])

# Save final dataset
df.to_csv("PowerBI_Final_Dataset.csv", index=False)

print("✅ Power BI dataset ready!")
