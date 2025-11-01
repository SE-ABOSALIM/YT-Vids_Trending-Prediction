import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('../Dataset/US_youtube_trending_data.csv')

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("crest")

print("=== DATASET SUMMARY ===")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("\nMissing Values:\n", df.isnull().sum()[df.isnull().sum() > 0])

# Gereksiz sütunları çıkar (isteğe bağlı)
df = df.drop(columns=["thumbnail_link", "description"], errors="ignore")

df["trending_date"] = pd.to_datetime(df["trending_date"], errors="coerce")
df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")

num_cols = ["view_count", "likes", "dislikes", "comment_count"]
print("\nNumerical Summary:\n", df[num_cols].describe().T)

for col in num_cols:
    plt.figure(figsize=(7,4))
    sns.histplot(df[col], bins=40, kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

corr = df[num_cols].corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Views, Likes, Dislikes, Comments")
plt.show()

top_videos = df.sort_values("view_count", ascending=False).head(10)
plt.figure(figsize=(10,6))
sns.barplot(y="title", x="view_count", data=top_videos)
plt.title("Top 10 Most Viewed Videos")
plt.xlabel("Views")
plt.ylabel("Video Title")
plt.show()

top_channels = df["channelTitle"].value_counts().head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=top_channels.values, y=top_channels.index)
plt.title("Top 10 Channels in Trending List")
plt.xlabel("Appearance Count")
plt.ylabel("Channel")
plt.show()

cat_views = df.groupby("categoryId")["view_count"].mean().sort_values(ascending=False)
plt.figure(figsize=(10,5))
sns.barplot(x=cat_views.index, y=cat_views.values)
plt.title("Average Views by Category ID")
plt.xlabel("Category ID")
plt.ylabel("Average Views")
plt.show()

plt.figure(figsize=(7,5))
sns.scatterplot(x="view_count", y="likes", data=df, alpha=0.4)
plt.xscale("log")
plt.yscale("log")
plt.title("Views vs Likes (Log Scale)")
plt.show()

df["like_ratio"] = df["likes"] / (df["dislikes"] + 1)
plt.figure(figsize=(8,4))
sns.histplot(df["like_ratio"], bins=50, kde=True)
plt.title("Like to Dislike Ratio Distribution")
plt.xlabel("Like Ratio")
plt.show()

trend_count = df["trending_date"].dt.date.value_counts().sort_index()
plt.figure(figsize=(12,5))
trend_count.plot(kind="line", marker="o")
plt.title("Number of Trending Videos Over Time")
plt.xlabel("Date")
plt.ylabel("Video Count")
plt.show()

for col in num_cols:
    plt.figure(figsize=(7,3))
    sns.boxplot(x=df[col])
    plt.title(f"Outlier Analysis for {col}")
    plt.show()

df["engagement_rate"] = (df["likes"] + df["comment_count"]) / df["view_count"]
eng = df[["title", "channelTitle", "view_count", "engagement_rate"]].sort_values("engagement_rate", ascending=False).head(10)

plt.figure(figsize=(10,5))
sns.barplot(x="engagement_rate", y="title", data=eng)
plt.title("Top 10 Videos by Engagement Rate")
plt.xlabel("Engagement Rate")
plt.ylabel("Video Title")
plt.show()

print("\n=== INSIGHTS SUMMARY ===")
print("✅ Views, likes, and comments are strongly correlated — viral videos show clear outliers.")
print("✅ Certain channels appear disproportionately often — possible loyal fanbase.")
print("✅ Some videos have unusually high engagement rates — strong audience interaction.")
print("✅ Like-to-dislike ratios vary widely, indicating audience polarization on some content.")
