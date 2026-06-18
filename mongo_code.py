from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os
from comment_class import RawComment
import pandas as pd

def mongo_connect():

  load_dotenv()
  uri = os.getenv('MONGODB_KEY')
  client = MongoClient(uri, server_api=ServerApi('1'))
  try:
      client.admin.command("ping")
  except ConnectionError as e:
      raise ConnectionError(f"Could not reach MongoDB at {uri}: {e}")

  print(f"connected to mongodb at {uri}")
  return client

def get_collection(db, collection_name):
   
   collection = db[collection_name]
   collection.create_index("video_id")
   return collection

def convert_df_to_list(df):
  raw_comments_list=[]
  records = df.to_dict("records")
  for row in records:
     comment = RawComment.df_to_comment(row)
     raw_comments_list.append(comment)
  
  return raw_comments_list

def insert_comments(collection, comments):
  if not comments:
    return  0
   
  if not all(isinstance(c, RawComment) for c in comments):
    raise TypeError("insert_raw_comments expects a list of RawComment objects")
  
  docs = [c.to_dict() for c in comments]
  
  result = collection.insert_many(docs, ordered=False)
  inserted = len(result.inserted_ids)

  print(f"inserted {inserted} new comment(s)")
  return inserted


def load_collection_to_df(collection):
  docs = list(collection.find({}))
  df = pd.DataFrame(docs)
  if "_id" in df.columns:
    df = df.rename(columns={"_id": "comment_id"})
  print(f"loaded {len(df)} documents from {collection.name}")
  return df