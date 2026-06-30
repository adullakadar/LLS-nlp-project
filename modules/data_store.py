'''
this file is mostly for any data-related functions such as saving or loading csv, loading df from csv, saving and loading transcripts.
this file also replaces mongocode.py as we shifted from mongodb to csv but essentially has the same functions

'''


import os
import pandas as pd
import json
import ast

# default directories to check and save, might change because the model_data folder is very cluttered
data_directory, trasncript_directory = 'model_data', 'model_data'

list_col = ['entities', 'keywords', 'noun_phrases']

# basic function to check if a csv file exists in the directories. used to check before commment extraction as a means to save api usage and reduce wastage.
def csv_exists(filename):
   return os.path.exists(os.path.join(data_directory, f"{filename}.csv"))

# function used to save a pd dataframe to csv, mostly used to save extracted comments into model_data foler
def save_df_csv(df, filename, drop_embedding= True):
  path = os.path.join(data_directory, f"{filename}.csv")

  output = df.copy()
  if drop_embedding and 'embedding' in output.columns: #this conditional is a safety measure for saving dfs. embeddings are always stored in faiss_store and cannot exist in any other file.
    output= output.drop(columns=['embedding']) #so for saving metadata df, embeddings is dropped and any other df being saved would not have embeddings yet.

  output.to_csv(path, index = False)
  print(f'save {filename} of {len(output)} cols  to {path}')

# function to load df, removes repeating code and saves api usage. v simple but had some problems with naming config, kept saving things with the wrong name like
# at first it wasn't csv, then it was namecsv.csv and like 2 more errors, partly blame my naming convention
def load_df_csv(filename):
    path = os.path.join(data_directory, f"{filename}.csv")
    df = pd.read_csv(path)
 
    for col in list_col:
        if col in df.columns:
            df[col] = df[col].apply(parse_column)
 
    print(f"loaded {filename} of {len(df)} rows from {path}")
    return df
# parses an individual column in a df
def parse_column(value):
   if isinstance(value, list):
      return value
   if pd.isna(value):
      return []
   try:
      parsed = ast.literal_eval(value)
      return parsed if isinstance(parsed, list) else []
   except (ValueError, SyntaxError):
      return []
 

def save_transcript(video_id, transcript_lines):
    path = os.path.join(trasncript_directory, f"{video_id}.json")
    with open(path, "w") as f:
        json.dump(transcript_lines, f, indent=2)
    print(f"saved {len(transcript_lines)} segments -> {path}")
 
 
def load_transcript(video_id):
    path = os.path.join(trasncript_directory, f"{video_id}.json")
    with open(path) as f:
        lines = json.load(f)
    print(f"loaded {len(lines)} segments from {path}")
    return lines
 
 
def transcript_exists(video_id):
    return os.path.exists(os.path.join(trasncript_directory, f"{video_id}.json"))