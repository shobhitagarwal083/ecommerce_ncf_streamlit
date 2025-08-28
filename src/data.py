import os
import io
import zipfile
import urllib.request
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)

ML100K_ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

def _download_movielens_100k():
    zip_path = os.path.join(DATA_DIR, "ml-100k.zip")
    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"{zip_path} not found. Please download manually from "
            "https://files.grouplens.org/datasets/movielens/ml-100k.zip "
            "and place it in the data/ folder."
        )
    return zip_path


def prepare_movielens_100k() -> pd.DataFrame:
    """Downloads & prepares MovieLens 100k interactions with titles.
    Returns a DataFrame with columns: user_id, item_id, rating, title
    """
    zip_path = _download_movielens_100k()
    with zipfile.ZipFile(zip_path, "r") as zf:
        # u.data -> user item rating timestamp (tab separated)
        with zf.open("ml-100k/u.data") as f:
            df = pd.read_csv(f, sep="\t", header=None, names=["user_id","item_id","rating","timestamp"])
        # u.item -> item_id | title | release | video | url | genres...
        with zf.open("ml-100k/u.item") as f:
            # This file is pipe-delimited and may contain non-utf8; handle encoding
            raw = f.read()
            s = raw.decode("latin-1")
            items = pd.read_csv(io.StringIO(s), sep="|", header=None, encoding="latin-1")
            items = items[[0,1]]
            items.columns = ["item_id","title"]
        df = df.merge(items, on="item_id", how="left")
        df = df[["user_id","item_id","rating","title"]]
    # Save a convenient CSV
    out_csv = DATA_DIR / "interactions.csv"
    df.to_csv(out_csv, index=False)
    print(f"Prepared interactions at {out_csv}")
    return df

def load_custom_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    need = {"user_id","item_id","rating"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"CSV must include columns: {need}, got {df.columns}")
    if "title" not in df.columns:
        df["title"] = df["item_id"].astype(str)
    return df[["user_id","item_id","rating","title"]]
