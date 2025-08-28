import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data import prepare_movielens_100k, DATA_DIR

from src.model import build_ncf

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

def _make_id_map(values):
    uniq = sorted(values.unique())
    idx_map = {v:i for i,v in enumerate(uniq)}
    return idx_map

def _encode_column(series, mapping):
    return series.map(mapping).astype(int)

def main():
    # 1) Load data (MovieLens 100k)
    df = prepare_movielens_100k()

    # 2) Map raw IDs -> contiguous ints for embeddings
    user_map = _make_id_map(df["user_id"])
    item_map = _make_id_map(df["item_id"])

    df["user_idx"] = _encode_column(df["user_id"], user_map)
    df["item_idx"] = _encode_column(df["item_id"], item_map)

    # 3) Train/val split
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    num_users = len(user_map)
    num_items = len(item_map)

    X_train = [train_df["user_idx"].values, train_df["item_idx"].values]
    y_train = train_df["rating"].values.astype("float32")
    X_val = [val_df["user_idx"].values, val_df["item_idx"].values]
    y_val = val_df["rating"].values.astype("float32")

    # 4) Build model
    model = build_ncf(num_users=num_users, num_items=num_items, emb_dim=32, hidden=64, dropout=0.2)

    # 5) Train
    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,  # keep small for quick demo; increase for better accuracy
        batch_size=1024,
        verbose=1
    )

    # 6) Save model
    model_path = ARTIFACTS / "model.h5"
    model.save(model_path)
    print(f"Saved model -> {model_path}")

    # 7) Save mappings
    # Convert NumPy int64 keys to str
    user_map_serializable = {str(int(k)): int(v) for k, v in user_map.items()}
    item_map_serializable = {str(int(k)): int(v) for k, v in item_map.items()}

    with open("artifacts/user_map.json", "w") as f:
        json.dump(user_map_serializable, f)

    with open("artifacts/item_map.json", "w") as f:
        json.dump(item_map_serializable, f)


    # 8) Save item titles
    df_items = df.drop_duplicates("item_id")[["item_id", "title"]]
    df_items.to_csv(ARTIFACTS / "item_titles.csv", index=False)

    # 9) Export learned embeddings for fast scoring
    #    (We can read these from the model's embedding layers)
    user_weights = model.get_layer("user_emb").get_weights()[0]
    item_weights = model.get_layer("item_emb").get_weights()[0]
    np.save(ARTIFACTS / "user_factors.npy", user_weights)
    np.save(ARTIFACTS / "item_factors.npy", item_weights)
    print("Saved user_factors.npy and item_factors.npy")

if __name__ == "__main__":
    main()
