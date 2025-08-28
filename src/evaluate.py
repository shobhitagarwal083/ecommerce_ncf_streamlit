import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load artifacts
MODEL_PATH = "artifacts/model.h5"
USER_MAP_PATH = "artifacts/user_map.json"
ITEM_MAP_PATH = "artifacts/item_map.json"
INTERACTIONS_PATH = "data/interactions.csv"

def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(USER_MAP_PATH, "r") as f:
        user_map = json.load(f)
        user_map = {int(k): v for k, v in user_map.items()}

    with open(ITEM_MAP_PATH, "r") as f:
        item_map = json.load(f)
        item_map = {int(k): v for k, v in item_map.items()}

    return model, user_map, item_map

def get_data(user_map, item_map):
    df = pd.read_csv(INTERACTIONS_PATH)

    user_col, item_col = "user_id", "item_id"

    df["user"] = df[user_col].map(user_map)
    df["item"] = df[item_col].map(item_map)

    df = df.dropna()

    # Cast only the required columns to int
    df[["user", "item", "rating"]] = df[["user", "item", "rating"]].astype(int)

    train, test = train_test_split(df, test_size=0.2, random_state=42)
    return train, test




def evaluate(model, train, test, top_k=10):
    users = test["user"].unique()
    hits, ndcg = [], []

    all_items = np.array(list(set(train["item"].unique()) | set(test["item"].unique())))

    for user in users:
        user_train_items = set(train[train["user"] == user]["item"])
        user_test_items = set(test[test["user"] == user]["item"])

        if not user_test_items:
            continue  # skip users with no test data

        # Candidate items = all items not seen in training
        candidates = list(set(all_items) - user_train_items)

        user_input = np.array([user] * len(candidates))
        item_input = np.array(candidates)

        scores = model.predict([user_input, item_input], verbose=0).flatten()
        top_items = [candidates[i] for i in np.argsort(scores)[::-1][:top_k]]

        # Compute HR@K
        hit = int(any(item in user_test_items for item in top_items))
        hits.append(hit)

        # Compute NDCG@K
        ndcg_score = 0
        for rank, item in enumerate(top_items):
            if item in user_test_items:
                ndcg_score = 1 / np.log2(rank + 2)
                break
        ndcg.append(ndcg_score)

    hr = np.mean(hits)
    ndcg = np.mean(ndcg)
    return hr, ndcg

if __name__ == "__main__":
    model, user_map, item_map = load_artifacts()
    train, test = get_data(user_map, item_map)

    hr, ndcg = evaluate(model, train, test, top_k=10)
    print(f"Hit Ratio@10: {hr:.4f}")
    print(f"NDCG@10: {ndcg:.4f}")
