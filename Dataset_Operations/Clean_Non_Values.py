import pandas as pd
import numpy as np

FILE_PATH = "../Dataset/Final_Dataset.csv"

df = pd.read_csv(FILE_PATH, dtype=object, low_memory=False)

required_cols = [
    "publishedAt",
    "channelTitle",
    "categoryId",
    "view_count",
    "likes",
    "comment_count"
]

null_like = {"None", "nan", "NaN", "<null>", "NULL", "", " ", "NoneType"}

def is_non_value(val):
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    s = str(val).strip()
    return s in null_like

mask = df[required_cols].applymap(is_non_value).any(axis=1)

to_delete = mask.sum()
print(f"🗑 Silinecek satır sayısı: {to_delete}")

df_cleaned = df[~mask]

df_cleaned.to_csv("dataset_cleaned.csv", index=False)

print("Non değerine sahip satırlar tamamen silindi.")
