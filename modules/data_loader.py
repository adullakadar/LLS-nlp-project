import ast
from pathlib import Path

import pandas as pd
import streamlit as st


META_FILE = Path("model_data/meta_data.csv")


TESLA_VIDEO_IDS = {
    "iueGI4CzP-0",
    "52O3cYsyZMo",
    "XTeWKmlNmN8",
    "KAJFALcJjac",
}


BYD_VIDEO_IDS = {
    # Add BYD video IDs here later
}


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

    if value in ["", "nan", "none", "null", "unknown"]:
        return "Unknown"

    return value.title()


def detect_brand_from_video_id(video_id):
    video_id = str(video_id).strip()

    if video_id in TESLA_VIDEO_IDS:
        return "tesla"

    if video_id in BYD_VIDEO_IDS:
        return "byd"

    return "unknown"


@st.cache_data
def load_all_data():
    df = pd.read_csv(META_FILE)

    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = [clean_col(col) for col in df.columns]

    if "source_type" in df.columns:
        comments_df = df[df["source_type"].astype(str).str.lower() == "comment"].copy()
    else:
        comments_df = df.copy()

    if "clean_text" not in comments_df.columns and "text" in comments_df.columns:
        comments_df["clean_text"] = comments_df["text"]

    if "text" not in comments_df.columns and "clean_text" in comments_df.columns:
        comments_df["text"] = comments_df["clean_text"]

    if "clean_text" not in comments_df.columns:
        comments_df["clean_text"] = ""

    if "text" not in comments_df.columns:
        comments_df["text"] = comments_df["clean_text"]

    if "sentiment" in comments_df.columns:
        comments_df["sentiment"] = comments_df["sentiment"].apply(fix_sentiment)
    else:
        comments_df["sentiment"] = "Unknown"

    if "sentiment_score" in comments_df.columns:
        comments_df["sentiment_score"] = pd.to_numeric(
            comments_df["sentiment_score"],
            errors="coerce"
        )
    else:
        comments_df["sentiment_score"] = None

    if "entities" in comments_df.columns:
        comments_df["entities"] = comments_df["entities"].apply(parse_list)
    else:
        comments_df["entities"] = [[] for _ in range(len(comments_df))]

    if "noun_phrases" in comments_df.columns:
        comments_df["noun_phrases"] = comments_df["noun_phrases"].apply(parse_list)
    else:
        comments_df["noun_phrases"] = [[] for _ in range(len(comments_df))]

    if "keywords" in comments_df.columns:
        comments_df["keywords"] = comments_df["keywords"].apply(parse_list)
    else:
        comments_df["keywords"] = [[] for _ in range(len(comments_df))]

    if "video_id" not in comments_df.columns:
        comments_df["video_id"] = "unknown_video"

    comments_df["video_id"] = comments_df["video_id"].astype(str)
    comments_df["brand"] = comments_df["video_id"].apply(detect_brand_from_video_id)

    if "serial_no" not in comments_df.columns:
        comments_df["serial_no"] = range(1, len(comments_df) + 1)

    return comments_df, df


@st.cache_resource
def load_rag():
    from modules.embeddings import embedding_setup
    from modules.retrieval import load_faiss
    from modules.singleagent import build_graph
    import modules.singleagent as agent

    embedding = embedding_setup()
    vs = load_faiss(embedding)

    agent.VECTORSTORE = vs

    graph = build_graph()

    return graph