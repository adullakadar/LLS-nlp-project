from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os
from comment_class import RawComment
import pandas as pd
from pymongo.errors import BulkWriteError, ConnectionFailure

def mongo_connect():

  load_dotenv()
  uri = os.getenv('MONGODB_KEY')
  client = MongoClient(uri, server_api=ServerApi('1'))
  try:
      client.admin.command("ping")
  except ConnectionFailure as e:
      raise ConnectionFailure(f"Could not reach MongoDB at {uri}: {e}")

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

def insert_df(collection, df):
  if df.empty:
      print("nothing to insert")
      return 0

  docs = df.to_dict("records")
  for doc in docs:
      # use comment_id as the Mongo primary key -> free dedup
      doc["_id"] = doc.pop("comment_id")

  try:
      result = collection.insert_many(docs, ordered=False)
      inserted = len(result.inserted_ids)
  except BulkWriteError as e:
      inserted = e.details.get("nInserted", 0)
      dups = len(e.details.get("writeErrors", []))
      print(f"skipped {dups} duplicate(s)")

  print(f"inserted {inserted} document(s) into {collection.name}")
  return inserted