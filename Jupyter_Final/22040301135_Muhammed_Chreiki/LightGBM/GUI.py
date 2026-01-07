import streamlit as st
import pandas as pd
import joblib
import numpy as np

CATEGORY_MAP = {
    "Film & Animation": 1,
    "Music": 10,
    "Pets & Animals": 15,
    "Sports": 17,
    "Travel & Events": 19,
    "Gaming": 20,
    "People & Blogs": 22,
    "Comedy": 23,
    "Entertainment": 24,
    "News & Politics": 25,
    "Howto & Style": 26,
    "Education": 27,
    "Science & Technology": 28,
    "Nonprofits & Activism": 29,
    "Movies": 30
}

def title_meta_features(series):
    return np.c_[
        series.str.len().fillna(0),
        series.str.split().apply(len).fillna(0),
        series.str.contains("!").astype(int),
        series.str.contains(r"\?").astype(int),
    ]

def hour_label(hour):
    if 5 <= hour < 12:
        return f"{hour:02d}:00 – {hour+1:02d}:00 (Morning)"
    elif 12 <= hour < 17:
        return f"{hour:02d}:00 – {hour+1:02d}:00 (Afternoon)"
    elif 17 <= hour < 21:
        return f"{hour:02d}:00 – {hour+1:02d}:00 (Evening)"
    else:
        return f"{hour:02d}:00 – {hour+1:02d}:00 (Night)"


model_pipeline = joblib.load("/Jupyter_Final/22040301135_Muhammed_Chreiki/model_pipeline.pkl")

st.set_page_config(page_title="YouTube Trending Predictor", layout="centered")

st.title("YouTube Trending Probability Predictor")
st.markdown("Video bilgilerini girerek, trend olma ihtimalini yüzdelik olarak anında gör.")

st.divider()

title = st.text_input(
    "Video Title",
    placeholder="Enter video title here..."
)

category_name = st.selectbox(
    "Video Category",
    options=list(CATEGORY_MAP.keys())
)
category_id = CATEGORY_MAP[category_name]

comments_disabled = st.radio(
    "Are comments disabled?",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

HOUR_OPTIONS = {hour_label(h): h for h in range(0, 24)}
hour_label_selected = st.selectbox(
    "Publish Time",
    options=list(HOUR_OPTIONS.keys())
)
publish_hour = HOUR_OPTIONS[hour_label_selected]

publish_day_of_week = st.selectbox(
    "Publish Day of Week",
    options=[1, 2, 3, 4, 5, 6, 7],
    format_func=lambda x: {
        1: "Monday",
        2: "Tuesday",
        3: "Wednesday",
        4: "Thursday",
        5: "Friday",
        6: "Saturday",
        7: "Sunday"
    }[x]
)

is_weekend = 1 if publish_day_of_week in [6, 7] else 0

st.info(
    "Weekend: Yes" if is_weekend else "Weekend: No"
)

if st.button("Predict Trending Probability"):
    if title.strip() == "":
        st.warning("Title cannot be empty.")
    else:
        input_df = pd.DataFrame([{
            "title": title,
            "categoryId": category_id,
            "comments_disabled": comments_disabled,
            "publish_hour": publish_hour,
            "publish_day_of_week": publish_day_of_week,
            "is_weekend": is_weekend
        }])

        prob = model_pipeline.predict_proba(input_df)[0, 1]

        if prob < 0.3:
            label = "🔵 Low chance"
        elif prob < 0.7:
            label = "🟡 Medium chance"
        else:
            label = "🔴 High chance"

        st.success(f"🔥 Trending Probability: **{prob * 100:.2f}%** ({label})")