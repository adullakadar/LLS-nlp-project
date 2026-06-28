import streamlit as st

from modules.data_loader import load_all_data
from modules.ui_pages import dashboard_page, explore_page


# =========================================================
# STREAMLIT CONFIG
# =========================================================

st.set_page_config(
    page_title="Tesla vs BYD YouTube Intelligence Engine",
    page_icon="🚗",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0F0E30 0%, #15143A 45%, #09091F 100%);
        color: white;
    }

    h1, h2, h3 {
        color: #FFFFFF;
        font-weight: 800;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    section[data-testid="stSidebar"] {
        background-color: #24242F;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.08);
        padding: 18px;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.12);
    }

    .comment-card {
        background: rgba(255,255,255,0.08);
        padding: 16px;
        border-radius: 14px;
        margin-bottom: 12px;
        border-left: 4px solid #8F8CFF;
    }

    .small-text {
        color: #C7C7E6;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD DATA
# =========================================================

comments_df, meta_df = load_all_data()

if comments_df.empty:
    st.error("No comment data found. Expected: model_data/tesla_comments_preprocessed.csv")
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🚗 EV NLP Engine")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Explore Comments"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")

brand_options = ["All"] + sorted(comments_df["brand"].dropna().unique().tolist())
sentiment_options = ["All"] + sorted(comments_df["sentiment"].dropna().unique().tolist())
video_options = ["All"] + sorted(comments_df["video_id"].dropna().astype(str).unique().tolist())

selected_brand = st.sidebar.selectbox("Brand", brand_options)
selected_sentiment = st.sidebar.selectbox("Sentiment", sentiment_options)
selected_video = st.sidebar.selectbox("Video ID", video_options)


# =========================================================
# APPLY FILTERS
# =========================================================

filtered_df = comments_df.copy()

if selected_brand != "All":
    filtered_df = filtered_df[filtered_df["brand"] == selected_brand]

if selected_sentiment != "All":
    filtered_df = filtered_df[filtered_df["sentiment"] == selected_sentiment]

if selected_video != "All":
    filtered_df = filtered_df[filtered_df["video_id"].astype(str) == selected_video]


# =========================================================
# HEADER
# =========================================================

st.title("Tesla vs BYD YouTube Intelligence Engine")
st.caption(
    "NLP dashboard for sentiment analysis, keyword extraction, named entity insights, and topic-style analysis."
)


# =========================================================
# PAGE ROUTING
# =========================================================

if page == "Dashboard":
    dashboard_page(filtered_df, meta_df)

elif page == "Explore Comments":
    explore_page(filtered_df)