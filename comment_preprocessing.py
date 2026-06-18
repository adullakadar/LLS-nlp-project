import pandas as pd
import re
import html


def preprocess_comment(text):
  text = html.unescape(text)

  # Remove URLs
  text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

  # Remove emails
  text = re.sub(r"\S+@\S+", " ", text)

  # Remove @mentions
  text = re.sub(r"@\w+", " ", text)

  # Remove hashtags symbol only, but keep the word
  text = re.sub(r"#", "", text)

  # Remove new lines and tabs
  text = re.sub(r"[\n\r\t]", " ", text)

  # Remove extra spaces
  text = re.sub(r"\s+", " ", text)

  # Strip spaces from start and end
  text = text.strip()
  
  return text

def preprocess_df(df):
  df_clean = df.copy()
  df_clean = df_clean.dropna(subset=["text"])

  df_clean["text"] = df_clean["text"].astype(str)

  df_clean = df_clean[df_clean["text"].str.strip() != ""]

  df_clean["clean_text"] = df_clean['text'].apply(preprocess_comment)

  df_clean = df_clean[df_clean["clean_text"].str.strip() != ""]
  df_clean = df_clean[df_clean["clean_text"].str.split().str.len() >= 3]

  df_clean = df_clean.drop_duplicates(subset=["clean_text"])
  df_clean = df_clean.reset_index(drop=True)
  print("Final number of comments after cleaning:", len(df_clean))
  print("Comments removed:", len(df) - len(df_clean))
  return df_clean