'''
topic modelling module
uses unsupervised lda to find topics in comments, grouping similar words together.
'''

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
 
# runs lda on a provided text column and returns model, the vectorizer and topic-related comments
def run_lda(df, text_col="clean_text", n_topics=10, max_features=1000):
  texts = df[text_col].dropna().astype(str)
  texts = texts[texts.str.strip() != ""]
#   word-count vectorizers, removes stopwords and keep somewhat uncommon topics.
  vectorizer = CountVectorizer(max_features=max_features, max_df=0.5, stop_words='english')
  X = vectorizer.fit_transform(texts)

  lda = LatentDirichletAllocation(
      n_components=n_topics,
      learning_method="batch",
      max_iter=25,
      random_state=0,
  )
  doc_topics = lda.fit_transform(X)

  print(f"fit LDA: {n_topics} topics over {X.shape[0]} docs, "
        f"{X.shape[1]} vocab terms")
  return lda, vectorizer, doc_topics

# returns words in comments associated strongly with a topic
def get_topic_words(lda, vectorizer, n_words=10):
    feature_names = np.array(vectorizer.get_feature_names_out())
    # sort weights by descending
    sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
 
    topics = []
    for topic_idx in range(lda.components_.shape[0]):
        top_words = feature_names[sorting[topic_idx, :n_words]]
        topics.append(list(top_words))
    return topics
 
 
def print_topics(lda, vectorizer, n_words=10):
    for i, words in enumerate(get_topic_words(lda, vectorizer, n_words)):
        print(f"Topic {i:>2}: {' '.join(words)}")

# finds topics for positive and negative sentiments. helps find topics associated with emotions like reliability being a negative or luxury being a positive
def topics_per_sentiment(df, text_col="clean_text", sentiment_col="sentiment",n_topics=5, n_words=8):
    results = {}
    for sentiment, group in df.groupby(sentiment_col):
        if len(group) < n_topics:        # too few docs to model
            print(f"skipping '{sentiment}': only {len(group)} comments")
            continue
        lda, vec, _ = run_lda(group, text_col, n_topics=n_topics)
        results[sentiment] = get_topic_words(lda, vec, n_words)
        print(f"\n--- topics for sentiment: {sentiment} ---")
        print_topics(lda, vec, n_words)
    return results
