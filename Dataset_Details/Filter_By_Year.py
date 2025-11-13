import pandas as pd

FILE_PATH = "../Dataset/dataset_cleaned.csv"
df = pd.read_csv(FILE_PATH)

df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
df["year"] = df["publishedAt"].dt.year
df = df[(df["year"] >= 2022) & (df["year"] <= 2025)]

trend_df = df[df["is_trending"] == 1]
nontrend_df = df[df["is_trending"] == 0]

trend_by_year = trend_df["year"].value_counts().sort_index()
nontrend_by_year = nontrend_df["year"].value_counts().sort_index()

print("\nTrend (2022–2025):")
print(trend_by_year)

print("\nNon-Trend (2022–2025):")
print(nontrend_by_year)
