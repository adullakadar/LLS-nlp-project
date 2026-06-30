import ast
from pathlib import Path

import pandas as pd
import streamlit as st

from modules.embeddings import embedding_setup
from modules.retrieval import load_faiss
from modules.singleagent import build_graph
import modules.singleagent as agent

TESLA_FILE = Path("model_data/tesla_comments_preprocessed.csv")
BYD_FILE = Path("model_data/byd_comments_preprocessed.csv")
META_FILE = Path("model_data/meta_data.csv")


def clean_col(col):
    return str(col).strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")


def parse_list(value):
    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    if isinstance(value, list):
        return value

    if isinstance(value, str):
        value = value.strip()

        if value == "" or value.lower() in ["nan", "none", "null"]:
            return []

        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else [value]
        except Exception:
            return [value]

    return []


def fix_sentiment(value):
    if value is None:
        return "Unknown"

    try:
        if pd.isna(value):
            return "Unknown"
    except Exception:
        pass

    value = str(value).strip().lower()

    if value in ["positive", "pos"]:
        return "Positive"

    if value in ["negative", "neg"]:
        return "Negative"

    if value in ["neutral", "neu"]:
        return "Neutral"

    return "Unknown" if value in ["", "nan", "none", "null", "unknown"] else value.title()


def load_comment_file(path, brand):
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = [clean_col(col) for col in df.columns]

    df["brand"] = brand
    df["source_type"] = "comment"

    if "clean_text" not in df.columns and "text" in df.columns:
        df["clean_text"] = df["text"]

    if "text" not in df.columns and "clean_text" in df.columns:
        df["text"] = df["clean_text"]

    if "clean_text" not in df.columns:
        df["clean_text"] = ""

    if "text" not in df.columns:
        df["text"] = df["clean_text"]

    df["sentiment"] = df["sentiment"].apply(fix_sentiment) if "sentiment" in df.columns else "Unknown"

    if "sentiment_score" in df.columns:
        df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
    else:
        df["sentiment_score"] = None

    df["entities"] = df["entities"].apply(parse_list) if "entities" in df.columns else [[] for _ in range(len(df))]
    df["keywords"] = df["keywords"].apply(parse_list) if "keywords" in df.columns else [[] for _ in range(len(df))]

    if "video_id" not in df.columns:
        df["video_id"] = "unknown_video"

    if "serial_no" not in df.columns:
        df["serial_no"] = range(1, len(df) + 1)

    return df


@st.cache_data
def load_all_data():
    tesla_df = load_comment_file(TESLA_FILE, "tesla")
    byd_df = load_comment_file(BYD_FILE, "byd")

    comments_df = pd.concat([tesla_df, byd_df], ignore_index=True)

    meta_df = pd.DataFrame()

    if META_FILE.exists():
        meta_df = pd.read_csv(META_FILE)
        meta_df = meta_df.loc[:, ~meta_df.columns.str.contains("^Unnamed")]
        meta_df.columns = [clean_col(col) for col in meta_df.columns]

    return comments_df, meta_df

def load_rag():
    embedding = embedding_setup()
    vs = load_faiss(embedding)
    agent.VECTORSTORE = vs
    graph = build_graph()

    return graph