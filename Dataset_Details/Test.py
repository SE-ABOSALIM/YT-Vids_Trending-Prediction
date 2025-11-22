import pandas as pd

FILE_PATH = "../Dataset/US_Videos_Dataset.csv"

df = pd.read_csv(FILE_PATH)

counts = df["is_trending"].value_counts().sort_index()

zero_count = counts.get(0, 0)
one_count = counts.get(1, 0)

print("0 sayısı:", zero_count)
print("1 sayısı:", one_count)
