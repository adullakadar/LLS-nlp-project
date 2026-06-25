from langchain_ollama import OllamaEmbeddings

def embedding_setup():
  embed_model = 'embeddinggemma'
  embedding = OllamaEmbeddings(model = embed_model)
  return embedding

def embed_text(text, embedding):
  return embedding.embed_documents(text)

def embed_df(df, embedding, text):
  df_emb = df.copy()
  vectors = embed_text(df_emb[text].tolist(), embedding)
  df_emb['embedding'] = vectors
  return df_emb
