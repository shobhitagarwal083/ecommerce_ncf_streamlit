import json
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse


# Load model
model = tf.keras.models.load_model("artifacts/model.h5")

# Load user & item maps
with open("artifacts/user_map.json", "r") as f:
    user_map = json.load(f)
with open("artifacts/item_map.json", "r") as f:
    item_map = json.load(f)

# Reverse map to get original IDs
reverse_item_map = {v: k for k, v in item_map.items()}

# Load movies.csv (adjust path if needed)
movies = pd.read_csv("data/ml-latest-small/movies.csv")
movie_dict = dict(zip(movies["movieId"].astype(str), movies["title"]))

def recommend(user_id, top_k=5):
    user_idx = user_map[str(user_id)]
    scores = model.predict([np.array([user_idx] * len(item_map)),
                            np.array(list(item_map.values()))],
                           verbose=0).flatten()

    top_indices = np.argsort(-scores)[:top_k]
    recommended_item_ids = [list(item_map.keys())[i] for i in top_indices]

    recommended_titles = []
    for item_id in recommended_item_ids:
        if item_id in movie_dict:
            recommended_titles.append(movie_dict[item_id])
        else:
            # Skip missing movies
            continue

    return recommended_titles



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference for a given user ID")
    parser.add_argument("--user", type=int, default=1, help="User ID to generate recommendations for")
    parser.add_argument("--top_k", type=int, default=5, help="Number of recommendations to return")
    args = parser.parse_args()

    print(f"Recommendations for user {args.user}:")
    for movie in recommend(args.user, top_k=args.top_k):
        print("-", movie)
