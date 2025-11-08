import pandas as pd

FILE_PATH = "../Dataset/US_youtube_trending_data_cleaned.csv"

df = pd.read_csv(FILE_PATH)
df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
df = df.dropna(subset=["publishedAt"])
df["year"] = df["publishedAt"].dt.year
df_range = df[(df["year"] >= 2022) & (df["year"] <= 2025)]
year_counts = df_range["year"].value_counts().sort_index()

print("\nVideo Count by Year:")
print(year_counts)
print("\nTotal (2022–2025):", len(df_range))
