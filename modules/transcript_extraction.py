import re
import nltk
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
import json

NP_CHUNK_GRAMMAR = r"NP: {<DT>?<JJ>*<NN.*>+}"
NP_CHINK_GRAMMAR = r"""
NP:
  {<.*>+}              # chunk everything first
  }<VB.*|IN|DT|CC>+{   # then chink out verbs, prepositions, determiners, conjunctions
"""

def nltk_setup():
  nltk.download("punkt")
  nltk.download("punkt_tab")
  nltk.download('averaged_perceptron_tagger')
  nltk.download('averaged_perceptron_tagger_eng')
  print("nltk packages downloaded")


def parse_transcript(video_id):

  youtube_api = YouTubeTranscriptApi()
  transcript = youtube_api.fetch(video_id, languages=["en"],).to_raw_data()
  
  transcript_lines = []
  for line in transcript:
    clean_line = preprocess_transcript_line(line['text'])
    if clean_line:
      transcript_lines.append({
        'text': clean_line,
        'start': line['start'],
        'duration': line['duration']
      })
  return transcript_lines

def save_transcript(video_id,transcript_lines):
    path = f'saved_transcripts/{video_id}.json'
    with open(path, 'w') as f:
       json.dump(transcript_lines,f,indent=2)

def load_transcript(video_id):
    path = f'saved_transcripts/{video_id}.json'
    with open(path) as f:
      return json.load(f)

def preprocess_transcript_line(line):

  line = re.sub(r"\[.*?\]", " ", line)
  line = re.sub(r"\s+", " ", line)
  line = line.strip()
  return line

def chunk_lines(transcript_lines, per_chunk_time):

  chunks = []
  current_text= []
  current_start = None

  for line in transcript_lines:
      if current_start is None:
        current_start = line['start']
      current_text.append(line['text'])

      elapsed = (line['start'] + line['duration']) - current_start
      if elapsed >= per_chunk_time:
         chunks.append({'start_time': current_start, 'text': ' '.join(current_text)})
         current_text = []
         current_start = None
  if current_text:
     chunks.append({'start_time':current_start, 'text': ' '.join(current_text)})
  return chunks

def phrases_from_tree(tree):
    
    phrases = []
    for subtree in tree.subtrees(filter=lambda t: t.label() == "NP"):
        words = [word for word, tag in subtree.leaves()]
        phrases.append(" ".join(words))
    return list(dict.fromkeys(phrases))   # de-dup, keep order
 
 
def extract_noun_phrases(text, grammar=NP_CHUNK_GRAMMAR):
    
    tagged = nltk.pos_tag(nltk.word_tokenize(text))
    tree = nltk.RegexpParser(grammar).parse(tagged)
    return phrases_from_tree(tree)
 
 
def extract_noun_phrases_chink(text, grammar=NP_CHINK_GRAMMAR):
    
    tagged = nltk.pos_tag(nltk.word_tokenize(text))
    tree = nltk.RegexpParser(grammar).parse(tagged)
    return phrases_from_tree(tree)

def extract_transcript(video_id, per_chunk_time):
  rows = []
  transcript_lines = parse_transcript(video_id)

  chunks = chunk_lines(transcript_lines, per_chunk_time)

  for i, chunk in enumerate(chunks):
      row = {
          "source_id": f"{video_id}_t_{i}",  # becomes Mongo _id, like comment source_id
          "source_type": "transcript",       # vs "comment" elsewhere
          "video_id": video_id,
          "chunk_index": i,
          "start_time": round(chunk["start_time"], 1),
          "text": chunk["text"],
      }
      row["noun_phrases"] = extract_noun_phrases(chunk["text"])
      rows.append(row)

  print(f"{video_id}: {len(chunks)} chunks")

  return pd.DataFrame(rows)