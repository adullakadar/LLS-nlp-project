import os
import pandas as pd
import json
import ast

data_directory, trasncript_directory = 'model_data', 'model_data'

list_col = ['entities', 'keywords', 'noun_phrases']

def csv_exists(file):
   return os.path.exists(os.path.join(data_directory, f'{file}.csv'))

def save_df_csv(df, filename, drop_embedding= True):
  path = os.path.join(data_directory,filename)

  output = df.copy()
  if drop_embedding and 'embedding' in output.columns:
    output= output.drop(columns=['embedding'])

  output.to_csv(path, index = False)
  print(f'save {filename} of {len(output)} cols  to {path}')

def load_df_csv(filename):
    path = os.path.join(data_directory, filename)
    df = pd.read_csv(path)
 
    for col in list_col:
        if col in df.columns:
            df[col] = df[col].apply(parse_column)
 
    print(f"loaded {filename} of {len(df)} rows from {path}")
    return df

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