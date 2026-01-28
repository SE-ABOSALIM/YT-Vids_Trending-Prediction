import streamlit as st
import pandas as pd
import joblib

# =====================
# CONFIG
# =====================
st.set_page_config(
    page_title="YouTube Trend Potential Predictor",
    layout="centered"
)

# =====================
# LOAD MODELS
# =====================
@st.cache_resource
def load_models():
    return {
        "XGBoost": joblib.load("./models/xgb_pipeline.pkl"),
        "LightGBM": joblib.load("./models/lgbm_pipeline.pkl"),
    }

models = load_models()

FEATURES = [
    "categoryId",
    "comments_disabled",
    "publish_hour",
    "publish_dayofweek",
    "is_weekend",
    "like_view_ratio",
    "comment_view_ratio",
    "engagement_score",
    "title_length",
    "description_length",
    "tag_count"
]

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

# =====================
# FEATURE ENGINEERING
# =====================
def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")

    df["publish_hour"] = df["publishedAt"].dt.hour
    df["publish_dayofweek"] = df["publishedAt"].dt.dayofweek
    df["is_weekend"] = df["publish_dayofweek"].isin([5, 6]).astype(int)

    df["like_view_ratio"] = df["likes"] / (df["view_count"] + 1)
    df["comment_view_ratio"] = df["comment_count"] / (df["view_count"] + 1)
    df["engagement_score"] = (df["likes"] + df["comment_count"]) / (df["view_count"] + 1)

    # Robust text handling (avoid None/NaN issues)
    df["title"] = df["title"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["tags"] = df["tags"].fillna("").astype(str)

    df["title_length"] = df["title"].str.len()
    df["description_length"] = df["description"].str.len()
    df["tag_count"] = df["tags"].apply(lambda x: 0 if x.strip() == "" else len(x.split("|")))

    # Ensure correct column order
    return df.reindex(columns=FEATURES)

# =====================
# UI
# =====================
st.title("🔥 YouTube Trend Potential Predictor")
st.markdown(
    "This tool estimates **how similar your video is to known trending videos**, "
    "based on engagement and metadata signals."
)

st.divider()

# Model selection
st.subheader("🧠 Model Selection")
model_name = st.selectbox("Select Model", ["XGBoost", "LightGBM"])
model = models[model_name]

st.divider()

# Engagement
view_count = st.number_input("View Count", min_value=0, step=100)
likes = st.number_input("Likes", min_value=0, step=10)
comment_count = st.number_input("Comment Count", min_value=0, step=5)

# Metadata
title = st.text_input("Video Title")
description = st.text_area("Video Description")
tags = st.text_input("Tags (separated by |)", value="")

category_name = st.selectbox("Category", list(CATEGORY_MAP.keys()))
category_id = CATEGORY_MAP[category_name]

comments_disabled = st.radio(
    "Are comments disabled?",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

publish_date = st.date_input("Publish Date")
publish_time = st.time_input("Publish Time")

publishedAt = pd.to_datetime(f"{publish_date} {publish_time}")

st.divider()

# =====================
# PREDICTION
# =====================
if st.button("Predict Trend Potential"):
    if title.strip() == "":
        st.warning("Title cannot be empty.")
    else:
        user_df = pd.DataFrame([{
            "view_count": view_count,
            "likes": likes,
            "comment_count": comment_count,
            "categoryId": category_id,
            "comments_disabled": comments_disabled,
            "publishedAt": publishedAt,
            "title": title,
            "description": description,
            "tags": tags
        }])

        final_input = prepare_input(user_df)

        prob = model.predict_proba(final_input)[0, 1]
        score = prob * 100

        if score < 30:
            label = "🔵 Low Trend Similarity"
        elif score < 70:
            label = "🟡 Medium Trend Similarity"
        else:
            label = "🔴 High Trend Similarity"

        st.success(f"📈 **Trend Potential: {score:.2f}%**")
        st.caption(f"Model used: {model_name}")
        st.markdown(f"**Assessment:** {label}")

        st.info(
            "⚠️ This score reflects similarity to previously trending videos "
            "based on engagement signals, not a guaranteed future outcome."
        )
