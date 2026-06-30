'''
module responsible for building and loading faiss vectors to and from faiss_store and semantic searches
'''


import os 
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# function used for loading collections from mongo, commented to remove confusion and is currently not being used

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


# builds faiss vectors out of embeddings column in metadata df. 
def build_faiss(texts, vectors, metadata, embeddings):
  index_path = 'faiss_store'
  text_embeddings = list(zip(texts, vectors))
  vectorstore = FAISS.from_embeddings(text_embeddings=text_embeddings, embedding=embeddings, metadatas=metadata)
  vectorstore.save_local(index_path)
  print(f"built + saved FAISS index ({len(texts)} vectors) -> {index_path}")
  return vectorstore
 
#  loads faiss vectors from faiss_store
def load_faiss(embeddings):
    index_path = 'faiss_store'
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

# performs a seach using faiss vectors
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

# function that replaces load_collection, instead of mongo collection it extracts from a df.
def load_corpus_from_df(df):
  texts = []
  vectors = []
  metadata = []
  for i, row in df.iterrows():
    text = row.get('clean_text')
    if not isinstance(text,str) or not text.strip():
      text = row.get('text')
    vector = row.get('embedding')
    
    if not isinstance(text, str) or not text.strip() or vector is None:
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

# performs a filtered semantic search, restricting by optional sourcetype and optional video_id.
def filtered_search(vectorstore, query, source_type=None, video_id=None,k=5, fetch_k=40000):

    # build the metadata filter from whatever was passed
    meta_filter = {}
    if source_type:
        meta_filter["source_type"] = source_type
    if video_id:
        meta_filter["video_id"] = video_id

    docs = vectorstore.similarity_search(
        query,
        k=k,
        fetch_k=fetch_k,
        filter=meta_filter if meta_filter else None,
    )

    return [{"text": d.page_content, **d.metadata} for d in docs]