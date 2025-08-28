import json
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from tensorflow.keras.models import load_model

ART = Path(__file__).resolve().parent / "artifacts"

st.set_page_config(page_title="E-commerce Recommender (NCF)", page_icon="🛒", layout="wide")
st.title("🛒 E-commerce Recommender (NCF)")
st.caption("MovieLens 100k used as a tiny stand-in for product interactions. Swap your dataset later!")

# Load artifacts (train first if missing)
required = ["model.h5", "user_map.json", "item_map.json", "item_titles.csv", "item_factors.npy", "user_factors.npy"]
missing = [f for f in required if not (ART / f).exists()]
if missing:
    st.error(f"Artifacts missing: {missing}. Please run `python -m src.train` locally first.")
    st.stop()

# Load artifacts
model = load_model(ART / "model.h5")
with open(ART / "user_map.json") as f:
    user_map = json.load(f)
with open(ART / "item_map.json") as f:
    item_map = json.load(f)
item_titles = pd.read_csv(ART / "item_titles.csv")
user_factors = np.load(ART / "user_factors.npy")
item_factors = np.load(ART / "item_factors.npy")

# Reverse maps
rev_user_map = {v:k for k,v in user_map.items()}
rev_item_map = {v:k for k,v in item_map.items()}

# UI controls
col1, col2, col3 = st.columns([2,2,1])
with col1:
    user_choices = list(user_map.keys())
    user_raw = st.selectbox("Select a user ID", options=user_choices)
with col2:
    topk = st.number_input("Top-N", min_value=3, max_value=50, value=10, step=1)
with col3:
    score_btn = st.button("Recommend")

def recommend_for_user(user_raw_id, k=10):
    # Lookup embedding index
    uidx = user_map[user_raw_id]
    uvec = user_factors[uidx]
    # Score all items by dot product with item factors
    scores = item_factors @ uvec
    # Exclude already interacted items for this user (optional, skipped for brevity)
    top_idx = np.argsort(-scores)[:k]
    item_ids = [rev_item_map[i] for i in top_idx]
    titles = item_titles[item_titles["item_id"].isin(item_ids)]["title"].tolist()
    return list(zip(item_ids, titles, scores[top_idx]))

if score_btn:
    st.subheader(f"Top-{topk} recommendations for user {user_raw}")
    recs = recommend_for_user(user_raw, k=topk)
    df = pd.DataFrame(recs, columns=["item_id","title","score"])
    st.dataframe(df, use_container_width=True)
    st.info("Tip: In production, store item vectors in an ANN index (Faiss/ScaNN) for millisecond retrieval.")

st.markdown("---")
with st.expander("How this works (Interview Notes)"):
    st.markdown(
        """
        - **Neural Collaborative Filtering (NCF)**: Learns embeddings for users/items from interactions (ratings).
        - **Serving**: After training, we extract the embedding matrices. For a user, compute dot products vs all items and pick Top-N.
        - **Scaling**: Replace brute-force scoring with an **Approximate Nearest Neighbor** index. Cache popular items. Incrementally update embeddings.
        - **Cold Start**: For brand new items/users, use content features (titles, text, images) or popularity priors until enough interactions accrue.
        """)
