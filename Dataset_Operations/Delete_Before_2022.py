import pandas as pd

FILE_PATH = "../Dataset/US_youtube_trending_data.csv"

df = pd.read_csv(FILE_PATH)
df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
df = df[df["publishedAt"].dt.year >= 2022]
df = df.drop(columns=[col for col in ["dislikes", "ratings_disabled"] if col in df.columns])
df["is_trending"] = 1
output_path = "../Dataset/US_youtube_trending_data_cleaned.csv"
df.to_csv(output_path, index=False)

print(f"S: {output_path}")
print(f"S: {len(df)}")
