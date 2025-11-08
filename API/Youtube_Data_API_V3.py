import requests
import pandas as pd
import random
import time
import os
from tqdm import tqdm
from datetime import datetime, timedelta

API_KEYS = []

OUTPUT_FILE = "../Dataset/non_trending_videos.csv"
MAX_RESULTS = 50
TARGET_COUNT = 100000
SAVE_INTERVAL = 1000

KEYWORDS = [
    "music", "movie", "vlog", "review", "funny", "gaming", "education", "sports",
    "tutorial", "travel", "science", "art", "food", "comedy", "documentary",
    "tech", "dance", "live", "shorts", "how to", "challenge", "reaction",
    "asmr", "interview", "test", "study", "ai", "robotics", "fashion", "nature"
]

if os.path.exists(OUTPUT_FILE):
    df_existing = pd.read_csv(OUTPUT_FILE)
    existing_ids = set(df_existing["video_id"].astype(str))
    print(f"Existing records loaded: {len(existing_ids)} entries\n")
else:
    existing_ids = set()
    print("New CSV will be created\n")

all_videos = []
total_count = 0
api_index = 0
quota_exhausted = [False] * len(API_KEYS)
page_token = None

def get_key():
    global api_index
    return API_KEYS[api_index % len(API_KEYS)]


def next_key():
    global api_index
    quota_exhausted[api_index] = True
    usable = [i for i, used in enumerate(quota_exhausted) if not used]

    if not usable:
        print("All API keys have reached their quota. Stopping data collection...")
        save_progress()
        exit(0)

    api_index = usable[0]
    print(f"Switched to next API key → {get_key()}")

def save_progress():
    global all_videos
    if not all_videos:
        return
    print(f"Saving {len(all_videos)} videos...")
    temp_df = pd.DataFrame(all_videos)
    mode = "a" if os.path.exists(OUTPUT_FILE) else "w"
    header = not os.path.exists(OUTPUT_FILE)
    temp_df.to_csv(OUTPUT_FILE, mode=mode, index=False, header=header)
    all_videos.clear()
    print(f"{OUTPUT_FILE} updated.")

def fetch_video_stats(video_ids):
    stats = {}
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        while True:
            url = (
                f"https://www.googleapis.com/youtube/v3/videos"
                f"?part=statistics,snippet&id={','.join(batch)}&key={get_key()}"
            )
            try:
                resp = requests.get(url, timeout=10)
                data = resp.json()

                if "error" in data:
                    msg = data["error"]["message"]
                    print(f"{msg}")
                    if "quota" in msg.lower():
                        next_key()
                        continue
                    else:
                        break

                for item in data.get("items", []):
                    vid = item["id"]
                    s = item.get("statistics", {})
                    snippet = item.get("snippet", {})
                    stats[vid] = {
                        "view_count": int(s.get("viewCount", 0)),
                        "likes": int(s.get("likeCount", 0)),
                        "comment_count": int(s.get("commentCount", 0)),
                        "categoryId": int(snippet.get("categoryId", 0)),
                        "tags": ", ".join(snippet.get("tags", [])),
                        "comments_disabled": 1 if "commentCount" not in s else 0,
                        "ratings_disabled": 1 if "likeCount" not in s else 0
                    }

                break
            except Exception as e:
                print("Error fetching video statistics:", e)
                next_key()
                time.sleep(1)
        time.sleep(0.15)
    return stats

def random_date_range_2022_2024():
    start = datetime(2022, 1, 1)
    end = datetime(2024, 12, 31)
    delta = end - start
    random_start = start + timedelta(days=random.randint(0, delta.days - 30))
    random_end = random_start + timedelta(days=30)
    return (
        random_start.strftime("%Y-%m-%dT00:00:00Z"),
        random_end.strftime("%Y-%m-%dT23:59:59Z"),
    )


print("Starting video collection (US region, 2022–2024 range)...\n")

for _ in tqdm(range(TARGET_COUNT // MAX_RESULTS * 2), desc="API requests"):
    query = random.choice(KEYWORDS)
    published_after, published_before = random_date_range_2022_2024()

    search_url = (
        f"https://www.googleapis.com/youtube/v3/search"
        f"?part=snippet&type=video&maxResults={MAX_RESULTS}"
        f"&regionCode=US&q={query}&videoDuration=any"
        f"&publishedAfter={published_after}&publishedBefore={published_before}"
        f"&key={get_key()}"
    )

    if page_token:
        search_url += f"&pageToken={page_token}"

    try:
        response = requests.get(search_url, timeout=10)
        data = response.json()
    except Exception as e:
        print("API request error:", e)
        next_key()
        continue

    if "error" in data:
        msg = data["error"]["message"]
        print(f"{msg}")
        if "quota" in msg.lower():
            next_key()
        continue

    page_token = data.get("nextPageToken", None)
    items = data.get("items", [])
    new_ids = []

    for item in items:
        if item["id"].get("kind") != "youtube#video":
            continue

        vid_id = item["id"].get("videoId")
        if not vid_id or vid_id in existing_ids:
            continue

        snippet = item["snippet"]
        all_videos.append({
            "video_id": vid_id,
            "title": snippet.get("title", ""),
            "publishedAt": snippet.get("publishedAt", ""),
            "channelId": snippet.get("channelId", ""),
            "channelTitle": snippet.get("channelTitle", ""),
            "description": snippet.get("description", ""),
            "thumbnail_link": snippet.get("thumbnails", {}).get("default", {}).get("url", ""),
            "is_trending": 0
        })
        existing_ids.add(vid_id)
        new_ids.append(vid_id)
        total_count += 1

    if new_ids:
        stats_data = fetch_video_stats(new_ids)
        for v in all_videos[-len(new_ids):]:
            vid = v["video_id"]
            if vid in stats_data:
                v.update(stats_data[vid])

    if total_count % SAVE_INTERVAL == 0 and total_count > 0:
        save_progress()
        print(f"Total videos collected: {total_count}\n")

    if total_count >= TARGET_COUNT:
        print("Target reached!")
        break

    time.sleep(0.25)

save_progress()
print(f"\nProcess completed. Total videos collected: {total_count}")
print(f"CSV file: {os.path.abspath(OUTPUT_FILE)}")
