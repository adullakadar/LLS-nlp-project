'''
this file handles embeddings in a df before being stored into faiss.
embeddings are used for semantic search, so it made sense to create embeddings in the df itself and not externally, even though doing it externally reduces a step.
we also tried using sentencetransformers before this but we either didn't know how to use it or it didn't provide good results
'''


from langchain_ollama import OllamaEmbeddings

# embeddinggemma setup, uses the model and returns the client but doesn't connect yet
def embedding_setup():
  embed_model = 'embeddinggemma'
  embed_client = OllamaEmbeddings(model = embed_model)
  return embed_client

# connects the embed_client to ollama and embeds a list of texts and returns its vectors.
# huge performance change was made here, before it used to process the entire list of texts and return vectors. this was doable in testing as we only used
# ~150 comments or so and it never failed, but with the whole comment section of videos spanning up to 10k, ollama would simply crash and return an error.
# so we limited the amount of embeddings per run so it wouldn't crash
def embed_text(texts, embed_client, batch_size=50):
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vectors.extend(embed_client.embed_documents(batch))
    return vectors

# embeds the text column in a provided df and adds an embedding column. again, this column is redundant later as it is only used to create faiss
# and is dropped when saving into a csv

def embed_df(df, embed_client, text):
  df_emb = df.copy()
  vectors = embed_text(df_emb[text].tolist(), embed_client)
  df_emb['embedding'] = vectors
  return df_emb
