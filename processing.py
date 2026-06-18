import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def spacy_setup():
  nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])
  return nlp

ALLOWED_ENTITY_LABELS = ["ORG", "PERSON", "GPE", "LOC", "PRODUCT", "NORP"]
 
BLOCKED_ENTITIES = {
    "sir", "madam", "bro", "dude", "lol", "lmao", "omg",
    "yes", "no", "ok", "okay",
    "n't", "n’t",
    "highkey", "lame",
    "stalks",
    "se yon si li", "si li",
    "focus st",
}
analyzer = SentimentIntensityAnalyzer()

def extract_entities_batch(texts, nlp, batch_size=100):
  all_entities = []
 
  for doc in nlp.pipe(texts, batch_size=batch_size):
    entities = []

    for ent in doc.ents:
      entity_text = ent.text.strip()

      if (ent.label_ in ALLOWED_ENTITY_LABELS
            and entity_text.lower() not in BLOCKED_ENTITIES
            and len(entity_text) > 2):
          entities.append(entity_text)

    # de-dup while preserving order
    entities = list(dict.fromkeys(entities))
    all_entities.append(entities)

  return all_entities

def extract_entities_df(df, nlp, text_col="clean_text"):
    """Add an `entities` column and return the df."""
    df = df.copy()
    df["entities"] = extract_entities_batch(df[text_col].tolist(), nlp)
    return df


def get_top_keywords(row, feature_names, top_n=5):
    """Top-N keywords for one TF-IDF row (sparse matrix row)."""
    row_data = row.toarray().flatten()
 
    if row_data.sum() == 0:
        return []
 
    top_indices = row_data.argsort()[-top_n:][::-1]
    keywords = feature_names[top_indices]
 
    # keep only keywords that actually have a non-zero weight
    keywords = [
        word for word in keywords
        if row_data[feature_names.tolist().index(word)] > 0
    ]
    return keywords

def extract_keywords_df(df, text_col="clean_text", top_n=5):
    """
    Fit TF-IDF over the corpus and add a `keywords` column (top-N per comment).
    Same vectorizer settings as the original: 1-2 grams, english stopwords,
    min_df=3, max 5000 features.
    """
    df = df.copy()
 
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
    )
 
    tfidf_matrix = vectorizer.fit_transform(df[text_col])
    feature_names = np.array(vectorizer.get_feature_names_out())
    print("TF-IDF matrix shape:", tfidf_matrix.shape)
 
    df["keywords"] = [
        get_top_keywords(tfidf_matrix[i], feature_names, top_n=top_n)
        for i in range(tfidf_matrix.shape[0])
    ]
    return df

 
def get_sentiment_score(text):
    return analyzer.polarity_scores(text)["compound"]
 
 
def get_sentiment_label(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"
 
 
def add_sentiment_df(df, text_col="clean_text"):
    """Add `sentiment_score` and `sentiment` columns and return the df."""
    df = df.copy()
    df["sentiment_score"] = df[text_col].apply(get_sentiment_score)
    df["sentiment"] = df["sentiment_score"].apply(get_sentiment_label)
    return df