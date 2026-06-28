from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd

API_KEY = "PASTE_API_KEY_HERE"

#option 1
'''
VIDEOS = {
    "before": ["bj7tdVNT30c", "IKebCANuirA", "dxv0NoIFboU", "8YJ3L8y6TZU", "XpFj9XfFcyE"],
    "after": ["8yiObBm6pT4", "BQzJTa4FKK4"]
}
'''

#option 2
VIDEOS = {
    "before": ["bj7tdVNT30c"],
    "after": ["8yiObBm6pT4"]
}

youtube = build("youtube", "v3", developerKey=API_KEY)

all_comments = []

for period, video_ids in VIDEOS.items():
    for video_id in video_ids:
        print(f"Trying {period} video: {video_id}")

        next_page_token = None

        try:
            while True:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=200,
                    pageToken=next_page_token,
                    textFormat="plainText"
                )

                response = request.execute()

                for item in response["items"]:
                    comment = item["snippet"]["topLevelComment"]["snippet"]

                    all_comments.append({
                        "period": period,
                        "video_id": video_id,
                        "comment": comment["textDisplay"],
                        "published_at": comment["publishedAt"],
                        "like_count": comment["likeCount"]
                    })

                next_page_token = response.get("nextPageToken")

                if not next_page_token:
                    break

        except HttpError as e:
            print(f"Skipped video {video_id}: {e}")
            continue

df = pd.DataFrame(all_comments)
df.to_csv("youtube_comments.csv", index=False)

print(f"Saved {len(df)} comments to youtube_comments.csv")
print(df["period"].value_counts())