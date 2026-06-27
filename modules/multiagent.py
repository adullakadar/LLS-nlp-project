"""
RAG router (LangGraph) -- built on Lab 8's 3_LangGraph_MultiAgent.py pattern.

Flow mirrors the lab's customer-service graph:
    classifier  -> route       -> retrieval agent -> answer agent -> END
    (lab: classify_email -> route_email -> billing/technical -> response_agent)

Here the classifier decides factual vs opinion, the router picks the matching
retrieval source (transcripts vs comments), and the answer agent generates the
final response from the retrieved chunks. MLflow autolog traces every step.

    pip install langgraph langchain-ollama mlflow
    ollama pull llama3.2:3b
"""

import mlflow
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

from modules.retrieval import filtered_search


# ----------------------------------------------------------------------
# MLflow + LLM (same setup as the lab)
# ----------------------------------------------------------------------
mlflow.set_experiment("RAG_Router")
mlflow.langchain.autolog()

llm = ChatOllama(model="qwen3:8b", temperature=0)


# ----------------------------------------------------------------------
# State -- carried through every node (lab's AgentState)
# ----------------------------------------------------------------------
class RAGState(TypedDict):
    query: str          # the user's question
    category: str       # "factual" or "opinion"
    chunks: list        # retrieved documents
    answer: str         # final generated answer


# ----------------------------------------------------------------------
# NODE 1 -- classifier (lab's classify_email)
# ----------------------------------------------------------------------
def classify_query(state: RAGState):
    prompt = f"""
    You are a query classifier for a system about electric cars (Tesla vs BYD).
    Query:
    {state['query']}
    Classify into ONE category only:
    - factual   (specs, features, price, range, how something works)
    - opinion   (what people think, comparisons of preference, sentiment)
    Return only the category name.
    """
    result = llm.invoke(prompt)
    category = result.content.strip().lower()
    return {"category": category}


# ----------------------------------------------------------------------
# ROUTER (lab's route_email)
# ----------------------------------------------------------------------
def route_query(state: RAGState):
    if "factual" in state["category"]:
        return "transcript_agent"
    else:
        return "comment_agent"


# ----------------------------------------------------------------------
# NODE 2a -- transcript retrieval (lab's technical_agent)
# ----------------------------------------------------------------------
def transcript_agent(state: RAGState):
    chunks = filtered_search(
        VECTORSTORE, state["query"], source_type="transcript", k=5
    )
    return {"chunks": chunks}


# ----------------------------------------------------------------------
# NODE 2b -- comment retrieval (lab's billing_agent)
# ----------------------------------------------------------------------
def comment_agent(state: RAGState):
    chunks = filtered_search(
        VECTORSTORE, state["query"], source_type="comment", k=5
    )
    return {"chunks": chunks}


# ----------------------------------------------------------------------
# NODE 3 -- answer generation (lab's response_agent)
# ----------------------------------------------------------------------
def answer_agent(state: RAGState):
    # stitch the retrieved chunks into a context block
    context = "\n".join(f"- {c['text']}" for c in state["chunks"])

    prompt = f"""
    Answer the user's question using ONLY the context below.
    If the context doesn't contain the answer, say you don't have enough information.

    Question:
    {state['query']}

    Context:
    {context}

    Give a concise, direct answer.
    """
    result = llm.invoke(prompt)
    return {"answer": result.content}


# ----------------------------------------------------------------------
# BUILD GRAPH (lab's builder pattern)
# ----------------------------------------------------------------------
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


# the vectorstore is set from the notebook before invoking (see usage below)
VECTORSTORE = None