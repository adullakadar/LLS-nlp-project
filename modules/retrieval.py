import os 
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# def load_collection(collection):
#   texts = []
#   vectors= []
#   metadata = []

#   for doc in collection.find({}):
#     text = doc.get('clean_text') or doc.get('text')
#     vector = doc.get('embedding')

#     if not text or not vector:
#       continue

#     texts.append(text)
#     vectors.append(vector)
#     metadata.append({
#       'source_id': doc.get('_id'),
#       'source_type': doc.get('source_type'),
#       'video_id': doc.get('video_id')
#     })
#   print(f'loaded {len(texts)} doc from collection {collection}')
#   return texts, vectors, metadata

def build_faiss(texts, vectors, metadata, embeddings):
  index_path = 'faiss_store'
  text_embeddings = list(zip(texts, vectors))
  vectorstore = FAISS.from_embeddings(text_embeddings=text_embeddings, embedding=embeddings, metadatas=metadata)
  vectorstore.save_local(index_path)
  print(f"built + saved FAISS index ({len(texts)} vectors) -> {index_path}")
  return vectorstore
 
 
def load_faiss(embeddings):
    index_path = 'faiss_store'
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)


def semantic_search(vectorestore, query, k=5, fetch_k = 20, lambda_mult = 0.5):
  retriever = vectorestore.as_retriever(
    search_type = 'mmr',
    search_kwargs = {
      'k':k,
      'fetch_k':fetch_k,
      'lambda_mult':lambda_mult
    }
  )
  docs = retriever.invoke(query)
  output = []
  for d in docs:
    output.append({'text': d.page_content, **d.metadata})
  return output


def load_corpus_from_df(df):
  texts = []
  vectors = []
  metadata = []
  for i, row in df.iterrows():
    text = row.get('clean_text') or row.get('text')
    vector = row.get('embedding')
    
    if not text or vector is None:
      continue

    texts.append(text)
    vectors.append(vector)
    metadata.append({
      'source_id': row.get('source_id'),
      'source_type':row.get('source_type'),
      'video_id':row.get('video_id')
    })

  print(f'loaded {len(texts)} from {df}')
  return texts,vectors, metadata
