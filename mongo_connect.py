from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os
from data_classes import RawComment

load_dotenv()
uri = os.getenv('MONGODB_KEY')
client = MongoClient(uri, server_api=ServerApi('1'))

youtube_db = client["youtube_data"]
raw_data = youtube_db["raw_data"]

def save_comment(comment: RawComment)-> None:
  doc = comment.to_dict()
  result = raw_data.update_one({"_id": comment.comment_id}, {"$set": doc}, upsert=True)