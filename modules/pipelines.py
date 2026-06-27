from modules.comment_extraction import extract_comments
from modules.comment_preprocessing import preprocess_df
from modules.processing import extract_entities_df, add_sentiment_df, extract_keywords_df
from modules.embeddings import embed_df

from modules.transcript_extraction import parse_transcript, extract_transcript
from modules.data_store import save_transcript, transcript_exists, save_df_csv, load_df_csv, csv_exists


# Comment extraction for 1 video, saves raw comments
def extract_video_comments_full(video_id, youtube_api, nlp, embedding, limit = 150):
  if csv_exists(f'{video_id}_raw.csv'):
    df = load_df_csv(f'{video_id}_raw.csv')
    print(f'loaded existing raw_file')
  else:
    df = extract_comments(youtube_api, video_id, limit=limit)
    print(f'extracted {len(df)} comments')
    save_df_csv(df, f'{video_id}_raw')
  df= preprocess_df(df)
  df = extract_entities_df(df,nlp,'clean_text')
  df=add_sentiment_df(df,'clean_text')
  df = extract_keywords_df(df, 'clean_text')
  df = embed_df(df ,embedding, 'clean_text')
  
  print(f'processed {len(df)} comments')
  return df

# transcript extraction for 1 video, saves raw transcript
# modify so it doesnt extract twice
def extract_video_transcript_full(video_id, nlp, embedding, win = 60):
  if not transcript_exists(video_id):
    transcript = parse_transcript(video_id)
    save_transcript(video_id,transcript)
    print(f'transcript does not exist, saving raw transcript')
  
  df = extract_transcript(video_id, win)
  df = extract_entities_df(df, nlp, 'text')
  df = embed_df(df, embedding, 'text')

  print(f'processed transcript for {video_id} of {len(df)}')
  return df