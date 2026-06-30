from modules.comment_extraction import extract_comments
from modules.comment_preprocessing import preprocess_df
from modules.processing import extract_entities_df, add_sentiment_df, extract_keywords_df
from modules.embeddings import embed_df
from modules.transcript_extraction import parse_transcript, extract_transcript, extract_transcript_from_lines
from modules.data_store import save_transcript, transcript_exists, save_df_csv, load_df_csv, csv_exists, load_transcript


# Comment extraction for 1 video, saves raw comments
# bugs: not saving as csv, doesnt detect csv, names it as csv.csv.
# FIXED ALL :thumbs_up:

def extract_video_comments_full(video_id, youtube_api, nlp, embedding, limit):
  raw_name = f'{video_id}_raw'
  if csv_exists(raw_name):
    print(f'loaded existing raw_file')
    df = load_df_csv(raw_name)
  else:
    print('raw file dont exist, extracting comments')
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
# modify so it doesnt extract twice -- DONE

def extract_video_transcript_full(video_id, nlp, embedding, win = 60):
  if not transcript_exists(video_id):
    transcript = parse_transcript(video_id)
    save_transcript(video_id,transcript)
    print(f'transcript does not exist, saving raw transcript')
  else:
    print('transcript exist,s loading transcript')
  lines = load_transcript(video_id)

  df = extract_transcript_from_lines(lines, video_id, win)
  df = extract_entities_df(df, nlp, 'text')
  df = embed_df(df, embedding, 'text')

  print(f'processed transcript for {video_id} of {len(df)}')
  return df