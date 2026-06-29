from collections import Counter

import pandas as pd
import streamlit as st

from modules.data_loader import parse_list


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_top_items(df, col, top_n=15):
    if df.empty or col not in df.columns:
        return pd.DataFrame(columns=["item", "count"])

    items = []

    for value in df[col]:
        parsed_items = parse_list(value)
        items.extend([str(item).strip() for item in parsed_items if str(item).strip()])

    return pd.DataFrame(
        Counter(items).most_common(top_n),
        columns=["item", "count"]
    )


def topic_insights(df):
    topics = {
        "Battery & Range": ["battery", "range", "charging", "charge", "kwh", "lfp"],
        "Interior & Comfort": ["interior", "seat", "comfort", "cabin", "screen"],
        "Software & Tech": ["software", "autopilot", "fsd", "carplay", "android"],
        "Price & Value": ["price", "cheap", "expensive", "value", "cost"],
        "Performance": ["speed", "fast", "performance", "acceleration", "drive"],
        "Reliability & Quality": ["reliable", "quality", "build", "service", "problem"],
    }

    rows = []

    for topic, words in topics.items():
        pattern = "|".join(words)

        matched = df[
            df["clean_text"]
            .astype(str)
            .str.contains(pattern, case=False, na=False)
        ]

        sentiment = "N/A"

        if len(matched) > 0 and "sentiment" in matched.columns:
            sentiment = matched["sentiment"].mode().iloc[0]

        rows.append({
            "Topic": topic,
            "Matched Comments": len(matched),
            "Keywords Used": ", ".join(words),
            "Most Common Sentiment": sentiment
        })

    return pd.DataFrame(rows)


def show_comment_cards(df):
    if df.empty:
        st.info("No comments found.")
        return

    for _, row in df.iterrows():
        st.markdown(
            f"""
            <div class="comment-card">
                <b>Brand:</b> {row.get("brand", "N/A")} |
                <b>Sentiment:</b> {row.get("sentiment", "N/A")} |
                <b>Video:</b> {row.get("video_id", "N/A")}
                <br><br>
                {row.get("clean_text", "")}
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# DASHBOARD PAGE
# =========================================================

def dashboard_page(df, meta_df):
    st.header("📊 Dashboard")

    total_comments = len(df)
    total_videos = df["video_id"].nunique() if "video_id" in df.columns else 0
    positive_count = len(df[df["sentiment"] == "Positive"]) if "sentiment" in df.columns else 0
    negative_count = len(df[df["sentiment"] == "Negative"]) if "sentiment" in df.columns else 0
    neutral_count = len(df[df["sentiment"] == "Neutral"]) if "sentiment" in df.columns else 0

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Comments", total_comments)
    col2.metric("Videos", total_videos)
    col3.metric("Positive", positive_count)
    col4.metric("Negative", negative_count)
    col5.metric("Neutral", neutral_count)

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.subheader("Sentiment Distribution")

        if "sentiment" in df.columns:
            sentiment_chart = df["sentiment"].value_counts().reset_index()
            sentiment_chart.columns = ["Sentiment", "Count"]
            st.bar_chart(sentiment_chart.set_index("Sentiment"))
        else:
            st.info("No sentiment column found.")

    with right:
        st.subheader("Brand Distribution")

        if "brand" in df.columns:
            brand_chart = df["brand"].value_counts().reset_index()
            brand_chart.columns = ["Brand", "Count"]
            st.bar_chart(brand_chart.set_index("Brand"))
        else:
            st.info("No brand column found.")

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.subheader("Top Keywords")

        top_keywords = get_top_items(df, "keywords")

        if top_keywords.empty:
            st.info("No keywords found.")
        else:
            st.bar_chart(top_keywords.set_index("item"))
            st.dataframe(top_keywords, use_container_width=True)

    with right:
        st.subheader("Top Named Entities")

        top_entities = get_top_items(df, "entities")

        if top_entities.empty:
            st.info("No entities found.")
        else:
            st.bar_chart(top_entities.set_index("item"))
            st.dataframe(top_entities, use_container_width=True)

    st.markdown("---")

    st.subheader("Topic-Style Insights")

    if "clean_text" in df.columns:
        st.dataframe(topic_insights(df), use_container_width=True)
    else:
        st.info("No clean_text column found for topic-style insights.")

    st.markdown("---")

    st.subheader("Metadata Overview")

    if meta_df.empty:
        st.info("No metadata file found. Optional file expected at model_data/meta_df.csv.")
    else:
        st.dataframe(meta_df.head(20), use_container_width=True)


# =========================================================
# EXPLORE COMMENTS PAGE
# =========================================================

def explore_page(df):
    st.header("🔍 Explore Comments")

    search = st.text_input(
        "Search comments",
        placeholder="battery, price, interior, software"
    )

    filtered = df.copy()

    if search.strip() and "clean_text" in filtered.columns:
        filtered = filtered[
            filtered["clean_text"]
            .astype(str)
            .str.contains(search, case=False, na=False)
        ]

    st.write(f"Showing **{len(filtered)}** comments.")

    display_cols = [
        "serial_no",
        "brand",
        "video_id",
        "clean_text",
        "sentiment_score",
        "sentiment",
        "entities",
        "keywords"
    ]

    display_cols = [col for col in display_cols if col in filtered.columns]

    if display_cols:
        st.dataframe(filtered[display_cols], use_container_width=True, height=500)
    else:
        st.dataframe(filtered, use_container_width=True, height=500)

    st.markdown("---")

    st.subheader("Comment Samples")

    sample_type = st.selectbox(
        "Sample type",
        ["Random", "Positive", "Negative", "Neutral"]
    )

    sample_df = filtered.copy()

    if sample_type != "Random" and "sentiment" in sample_df.columns:
        sample_df = sample_df[sample_df["sentiment"] == sample_type]

    if len(sample_df) > 0:
        sample_rows = sample_df.sample(min(5, len(sample_df)), random_state=42)
        show_comment_cards(sample_rows)
    else:
        st.info("No comments found for this sample type.")

def rag_page(graph):
    st.header("🤖 Ask the Engine")
    st.caption(
        "Ask a question about Tesla vs BYD. Factual questions are answered from "
        "video transcripts; opinion questions from viewer comments."
    )
 
    query = st.text_input(
        "Your question",
        placeholder="e.g. what is the battery capacity?  /  is tesla better than byd?"
    )
 
    ask = st.button("Ask")
 
    if ask and query.strip():
        with st.spinner("Thinking... (classifying, retrieving, answering)"):
            result = graph.invoke({"query": query})
 
        category = result.get("category", "unknown")
        answer = result.get("answer", "")
        chunks = result.get("chunks", [])
 
        # routed category
        source_label = "📄 Transcripts" if "factual" in category else "💬 Comments"
        st.markdown(
            f"**Query type:** `{category}`  →  retrieved from **{source_label}**"
        )
 
        # the answer
        st.subheader("Answer")
        st.markdown(
            f"""
            <div class="comment-card">
                {answer}
            </div>
            """,
            unsafe_allow_html=True
        )
 
        # the retrieved context (so grounding is visible)
        st.subheader("Retrieved Sources")
        if not chunks:
            st.info("No source chunks retrieved.")
        else:
            for i, c in enumerate(chunks, start=1):
                st.markdown(
                    f"""
                    <div class="comment-card">
                        <span class="small-text">
                            Source {i} &middot; {c.get('source_type', 'N/A')} &middot;
                            video {c.get('video_id', 'N/A')}
                        </span>
                        <br><br>
                        {c.get('text', '')}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
 
    elif ask:
        st.warning("Please enter a question.")