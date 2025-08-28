# E-commerce Recommender (Neural Collaborative Filtering + Streamlit)

A fast, interview-ready project you can build in 2–3 days. It trains a **Neural Collaborative Filtering (NCF)** model on the **MovieLens 100k** dataset (as a stand-in for e-commerce interactions) and deploys a simple **Streamlit** app that serves **Top-N product (movie) recommendations** for a selected user.

> Why MovieLens? It’s tiny, clean, and downloads in seconds. You can swap in any user–item–rating dataset later (e.g., Amazon reviews subset) using the same pipeline.

---

## 1) Setup (one-time)

### Option A: Conda
```bash
conda create -n ncf python=3.10 -y
conda activate ncf
pip install -r requirements.txt
```

### Option B: venv
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2) Train locally (downloads MovieLens automatically)
```bash
# From project root
python -m src.train
```
This will:
- Download and prepare MovieLens 100k to `data/`
- Train an NCF model
- Export artifacts (model + mappings + embeddings) to `artifacts/`

Artifacts saved:
```
artifacts/
  model.h5
  user_map.json
  item_map.json
  item_titles.csv
  item_factors.npy
  user_factors.npy
```

---

## 3) Run the Streamlit app
```bash
streamlit run app_streamlit.py
```
Open the local URL Streamlit prints (usually http://localhost:8501). Pick a user and see Top-N recommendations.

---

## 4) Deploy (quick options)

### A) Streamlit Community Cloud
1. Push this folder to a public GitHub repo.
2. Create a new app on Streamlit Cloud, point it to your repo.
3. Set **Main file**: `app_streamlit.py`
4. Add `requirements.txt`. First run will train and cache model artifacts locally if missing.

### B) Hugging Face Spaces (Gradio/Streamlit)
1. Create a new Space → **Streamlit** template.
2. Upload the project files (or connect GitHub).
3. On first build, it will install requirements and run. You can pre-train locally and commit the `artifacts/` folder to the repo for faster start.

---

## 5) Talk Track for Interviews (Amazon SDE 1)

- **Customer Obsession:** Personalized suggestions reduce friction and discovery time.
- **Invent & Simplify:** Used concise NCF architecture with embeddings to learn user/item representations.
- **Scalability:** In production, store item embeddings in an ANN index (Faiss / ScaNN), cache hot items, batch offline training + online serving.
- **Ownership / Deliver Results:** End-to-end: data prep → training → serving → UI → deploy.

---

## 6) Swap in an Amazon-style dataset later
If you have a CSV with columns: `user_id,item_id,rating,title(optional)` you can place it in `data/custom_interactions.csv` and modify `src/data.py` → `load_custom_csv(...)` to train on it (no code redesign required).

---

## Repo Structure
```
.
├── app_streamlit.py
├── requirements.txt
├── README.md
├── src
│   ├── data.py
│   ├── model.py
│   └── train.py
└── artifacts
```
# ecommerce_ncf_streamlit_1
# ecommerce_ncf_streamlit
# ecommerce_ncf_streamlit1
# ecommerce_ncf_streamlit1
# ecommerce_ncf_streamlit1
# ecommerce_ncf_streamlit
