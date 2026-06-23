from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os
import pandas as pd
from pymongo.errors import BulkWriteError, ConnectionFailure

def mongo_connect():

  load_dotenv()
  uri = os.getenv('MONGODB_KEY')
  client = MongoClient(uri, server_api=ServerApi('1'))
  try:
      client.admin.command("ping")
  except ConnectionFailure as e:
      raise ConnectionFailure(f"Could not reach MongoDB, error: {e}")

  print(f"connected to mongodb")
  return client

def get_collection(db, collection_name):
   collection = db[collection_name]
   collection.create_index("video_id")
   return collection

def load_collection_to_df(collection):
  docs = list(collection.find({}))
  df = pd.DataFrame(docs)
  if "_id" in df.columns:
    df = df.rename(columns=["_id"])
  print(f"loaded {len(df)} documents from {collection.name}")
  return df

def insert_df(collection, df):
  
  if df.empty:
      print("nothing to insert")
      return 0

  docs = df.to_dict("records")
  for doc in docs:
      if "source_id" not in doc:
         raise KeyError('no source_id in row, stopping insertion otherwise datapoint loses its unique id')
      doc["_id"] = doc.pop("source_id")

  try:
      result = collection.insert_many(docs, ordered=False)
      inserted = len(result.inserted_ids)
  except BulkWriteError as e:
      inserted = e.details.get("nInserted", 0)
      dups = len(e.details.get("writeErrors", []))
      print(f"skipped {dups} duplicate(s)")

  print(f"inserted {inserted} document(s) into {collection.name}")
  return inserted