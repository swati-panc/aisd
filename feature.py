
import pandas as pd

# 1. Create raw data
data = {
    "Name": [" Rahul ", "Priya", "Amit", "Neha"],
    "Marks": [85, 32, 76, 90],
    "Attendance": [90, 60, 80, 95]
}

df = pd.DataFrame(data)

print("Raw Data:")
print(df)

# 2. Clean data
df["Name"] = df["Name"].str.strip()
df = df.fillna(0)

# 3. Feature engineering
df["Performance"] = df["Marks"] / 100
df["Attendance_Rate"] = df["Attendance"] / 100

df["Result"] = df["Marks"].apply(
    lambda x: 1 if x >= 40 else 0
)

# 4. Select model-ready features
features = df[[
    "Marks",
    "Attendance_Rate",
    "Performance",
    "Result"
]]

print("\nModel-Ready Features:")
print(features)

# 5. Save processed data
features.to_csv("features.csv", index=False)

print("\nFeature Engineering Pipeline Completed!")
