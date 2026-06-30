"""
this file orchestrates the answering agent when used in app.py
it takes a user question and runs it using langgraph to come up with a coherent answer.
the first step is to classify the question. this step was learning while building the agent, where the agent would not recognize whether the question asked for an opinion or a fact.
after classifying the question, it routes to the correct source type based on the question, facts come from transcripts and opinons from comments.
after retrieving its source, it uses it to answer the question.
"""

import mlflow
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

from modules.retrieval import filtered_search

VECTORSTORE = None

# monitoring step
mlflow.set_experiment('RAG_Router')
mlflow.langchain.autolog()

# we used qwen3 for this project. We tried using llama3.2 before this and it worked for simple one line queries but failed in performance for harder queries, taking 5-10 sometimes
# up to 15 minutes to answer a couple questions. Qwen is slower but answers harder questions and provides better answers.
# qwen handles classifying the question and answering and we used a temp=0 for deterministic approach. will try experimenting with temperature after and see how results vary.
llm = ChatOllama(model ='qwen3:8b', temperature= 0)


# ragstate that each node uses. query - user's question, category - classified question, chunks - retrieved files, answer - generated answer
class RAGState(TypedDict):
    query: str
    category: str
    chunks: list
    answer: str

# first node: classify question
# qwen labels the question based on fact or opinion. the determined answer will reflect in state and passed to further nodes.
def classify_query(state: RAGState):

    prompt = f'''
    You are a query classifier for a system about electric cars (Tesla vs BYD).
    Query:
    {state['query']}
    Classify into ONE category only:
    - factual   (specs, features, price, range, how something works)
    - opinion   (what people think, comparisons of preference, sentiment)
    Return only the category name.
    '''
    result= llm.invoke(prompt)
    category= result.content.strip().lower()

    return {'category':category}


# routes query based on result from classify_query, if factual go to transcript_agent, if opinion go to comment_agent
def route_query(state: RAGState):
    if 'factual' in state['category']:
        return 'transcript_agent'
    else:
        return 'comment_agent'

# second node: fetch transcripts
# if route_query returned transcript_agent this runs. it does a filtered search using embeddings only for transcript chunks.
def transcript_agent(state:RAGState):
    chunks= filtered_search(VECTORSTORE, state['query'], source_type='transcript', k=5)
    return {'chunks': chunks}


# second node: fetch comments
# same as transcript_agent but for comments
def comment_agent(state: RAGState):
    chunks = filtered_search(VECTORSTORE, state['query'], source_type='comment', k=5)
    return {'chunks': chunks}


# third node: generate answer
# generates an answer based on the retrieved chunks (gets joined into one variable).
# factual and opinion have different prompts, factual wants a concrete answer where opinion always tries to summarize collected information.
def answer_agent(state: RAGState):
    context = "\n".join(f"- {c['text']}" for c in state['chunks'])

    if 'factual' in state['category']:
        prompt = f"""
        Answer the question using the context below (from video reviews).
        The context is reliable — extract the relevant facts and answer directly.
        Only say you lack information if the context truly contains nothing relevant.

        Question: {state['query']}
        Context: {context}
        Answer:
        """
    else:  # opinion
        prompt = f"""
        The context below is viewer comments. Summarize what people think about
        the question — what they praise, criticize, and disagree on. Do NOT look
        for a single objective answer; characterize the range of opinions and
        common themes. Be specific about the sentiments expressed.

        Question: {state['query']}
        Comments: {context}
        Summary of opinions:
        """

    result = llm.invoke(prompt)
    return {"answer": result.content}


# langgraph
# entry -> conditional edge -> answer
# 
def build_graph():
    builder = StateGraph(RAGState)

    builder.add_node("classifier", classify_query)
    builder.add_node("transcript_agent", transcript_agent)
    builder.add_node("comment_agent", comment_agent)
    builder.add_node("answer_agent", answer_agent)

    builder.set_entry_point("classifier")
    builder.add_conditional_edges(
        "classifier",
        route_query,
        {
            "transcript_agent": "transcript_agent",
            "comment_agent": "comment_agent",
        },
    )
    builder.add_edge("transcript_agent", "answer_agent")
    builder.add_edge("comment_agent", "answer_agent")
    builder.add_edge("answer_agent", END)

    return builder.compile()
