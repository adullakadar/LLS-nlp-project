'''
currently not being used, mongo was the original database system we would use for this. There was no fault with it, even though it added complexity it was a nicer alternative to
csv and it provided more useful features.
we stopped using mongo because of its reproducibility factor, where the user would have to first install mongo and second, get a collection string from it. It adds to installation
comeplxity and we didn't want that for submission so we decided to just stick to csv as we've used that during the labs with no problems.
'''

from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os
import pandas as pd
from pymongo.errors import BulkWriteError, ConnectionFailure

# loads mongo client using mongo uri
def mongo_connect():

  load_dotenv()
  uri = os.getenv('MONGODB_KEY')
  mongo_client = MongoClient(uri, server_api=ServerApi('1'))
  try:
      mongo_client.admin.command("ping")
  except ConnectionFailure as e:
      raise ConnectionFailure(f"Could not reach MongoDB, error: {e}")

  print(f"connected to mongodb")
  return mongo_client

# returns a collection
def get_collection(db, collection_name):
   collection = db[collection_name]
   collection.create_index("video_id")
   return collection

# returns a df made from a mongo collection
def load_collection_to_df(collection,):
    docs = list(collection.find({}))
    df = pd.DataFrame(docs)
    if "_id" in df.columns:
        df = df.rename(columns={"_id": 'source_id'})
    print(f"loaded {len(df)} documents from {collection.name}")
    return df

# inserts a df into mongo collection, the _id would become source_id. source_id is the specific id associated with a source type, like each comment has its own id and
# transcript chunks would have their own id.
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