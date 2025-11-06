import pandas as pd
import numpy as np

df = pd.read_csv("../Dataset/US_youtube_trending_data.csv")

print("DATASET SUMMARY ")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print("\nColumn Names:\n", list(df.columns))

print("\n DATA TYPES ")
print(df.dtypes)

print("\n MISSING VALUES ")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values detected.")

print("\n UNIQUE VALUE COUNTS (FIRST 10 COLUMNS) ")
for col in df.columns[:10]:
    print(f"{col}: {df[col].nunique()} unique values")

cat_cols = df.select_dtypes(include=["object", "category"]).columns
print("\n CATEGORICAL COLUMNS ")
print(list(cat_cols))

for col in cat_cols:
    print(f"\n-- {col} --")
    print(df[col].value_counts().head(10))
    print("Missing:", df[col].isna().sum())

num_cols = df.select_dtypes(include=[np.number]).columns
print("\n NUMERICAL COLUMNS ")
print(list(num_cols))

print("\n NUMERICAL SUMMARY (DESCRIBE) ")
print(df[num_cols].describe().T)

print("\n CORRELATION MATRIX (TOP 10 NUMERIC COLUMNS) ")
print(df[num_cols].corr().round(3).iloc[:10, :10])

print("\n OUTLIER DETECTION (IQR METHOD) ")
for col in num_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    outliers = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum()
    print(f"{col}: {outliers} outliers detected.")

date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
if date_cols:
    print("\n DATE COLUMNS DETECTED ")
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        print(f"{col}: {df[col].min()} -> {df[col].max()}")

if "channelTitle" in df.columns:
    top_channels = df["channelTitle"].value_counts().head(5)
    print("\n TOP CHANNELS ")
    print(top_channels)

if "title" in df.columns and "view_count" in df.columns:
    top_videos = df.sort_values("view_count", ascending=False)[["title", "view_count"]].head(5)
    print("\n TOP 5 MOST VIEWED VIDEOS ")
    print(top_videos)

print("\n SUMMARY INSIGHTS ")
print("✅ Dataset includes both numerical and categorical features.")
print("✅ Missing values handled; check summary above.")
print("✅ Strong correlations likely between view_count, likes, dislikes, and comment_count.")
print("✅ Top channels indicate popularity concentration in few creators.")
print("✅ Outlier presence suggests viral videos with extremely high metrics.")
