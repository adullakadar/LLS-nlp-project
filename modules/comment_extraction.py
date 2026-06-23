import googleapiclient.discovery
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

dev = os.getenv("YT_API_KEY")

def api_setup():
  api_service_name = "youtube"
  api_version = "v3"
  DEVELOPER_KEY = dev
  youtube_api = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY)
  print("loaded youtube api")
  return youtube_api


def top_level_format(item, video_id):
    
    snippet = item["snippet"]["topLevelComment"]["snippet"]
    return {
        "source_id": item["snippet"]["topLevelComment"]["id"],
        "source_type": "comment",
        "video_id": video_id,
        "parent_id": None,
        "is_reply": False,
        "author": snippet["authorDisplayName"],
        "text": snippet["textOriginal"],
        "like_count": snippet["likeCount"],
        "public": item["snippet"]["isPublic"]   
    }
 
 
def reply_format(reply, video_id):
    
    snippet = reply["snippet"]
    return {
        "source_id": reply["id"],
        "source_type": "comment",
        "video_id": video_id,
        "parent_id": snippet["parentId"],
        "is_reply": True,
        "author": snippet["authorDisplayName"],
        "text": snippet["textOriginal"],
        "like_count": snippet["likeCount"],
        "public": True
    }


def parse_top_comments(video_id, youtube_api, limit):
    rows = []
    next_page_token = None
 
    while True:
        request = youtube_api.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token,
            textFormat="plainText",
        )
        response = request.execute()

        for item in response.get("items", []):
            rows.append(top_level_format(item, video_id))

            if item["snippet"].get("totalReplyCount", 0) > 0:
                parent_id = item["snippet"]["topLevelComment"]["id"]
                rows.extend(parse_replies(parent_id, video_id, youtube_api))
 
        print(f"Video {video_id}: collected {len(rows)} rows so far")
        
        if limit and len(rows) >= limit:
            break
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
 
    return rows

def parse_replies(parent_id, video_id, youtube_api):
    
    replies = []
    next_page_token = None
 
    while True:
        request = youtube_api.comments().list(
            part="snippet",
            parentId=parent_id,
            maxResults=100,
            pageToken=next_page_token,
        )
        response = request.execute()
 
        for reply in response.get("items", []):
            replies.append(reply_format(reply, video_id))
 
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
    return replies



def extract_comments(youtube_api, video_id,limit=100):

  comments = []

  comments.extend(
      parse_top_comments(video_id, youtube_api,limit)
  )

  df = pd.DataFrame(comments)
  df = df.drop_duplicates(subset=['source_id'])
  df = df.dropna(subset=['text'])
  df = df[df["text"].str.strip() != ""]
  df = df.reset_index(drop=True)

  return df

 