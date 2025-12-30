import os
import json
from pathlib import Path
import streamlit as st
import pandas as pd
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from bm25s.utils.beir import BASE_URL
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Smart Document Semantic Search Engine",
    layout="wide",
    page_icon="🔍"
)

# -----------------------------------------------------------
# WEBSITE-LIKE THEME / CSS (navbar + cards + centered container)
# Adjusted padding/margins and z-index so navbar is fully visible
# -----------------------------------------------------------
st.markdown(
    """
    <style>
    /* ensure main container sits below Streamlit top bar and navbar is visible */
    .block-container {
      max-width: 1100px;
      margin-left: auto;
      margin-right: auto;
      padding-top: 70px; /* increased to avoid overlap with Streamlit header */
      padding-bottom: 48px;
    }

    /* Navbar */
    .top-nav {
        background: linear-gradient(90deg,#0f4db1,#1f6bd6);
        color: white;
        padding: 18px 26px;
        border-radius: 8px;
        margin-bottom: 18px;
        font-weight: 700;
        font-size: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 18px rgba(15,77,177,0.18);
        position: relative;
        z-index: 9999;
        margin-top: 8px; /* push down slightly to ensure full visibility */
    }
    .top-nav .brand { display:flex; align-items:center; gap:12px; }
    .brand img { height:36px; border-radius:6px; }

    /* Content cards */
    .card {
      background-color: #ffffff;
      padding: 16px 18px;
      border-radius: 10px;
      margin-bottom: 14px;
      border: 1px solid #e9eef5;
      box-shadow: 0px 6px 20px rgba(18,44,82,0.04);
    }

    /* Options row look */
    .options-row .stSelectbox, .options-row .stNumberInput {
      display:inline-block;
      vertical-align: middle;
      margin-right: 12px;
    }

    /* Search box styling (makes textarea visually larger) */
    div[data-baseweb="input"] > input, textarea {
      border-radius: 8px !important;
      border: 1px solid #d6e0f0 !important;
      padding: 10px 12px !important;
      box-shadow: none !important;
      height: 44px !important;
    }

    /* Headings */
    h1, h2 { color:#0b2f66; }
    h2 { margin-top: 8px; margin-bottom: 6px; }

    /* Badges and score */
    .score-badge {
        background-color: #0f4db1;
        color: #fff;
        padding: 4px 9px;
        border-radius: 8px;
        font-size: 13px;
        font-weight:600;
    }
    .method-badge {
        font-size: 12px;
        font-weight: 700;
        padding: 6px 10px;
        color: white;
        border-radius: 8px;
    }
    .method-bm25 { background-color: #0078d4; }
    .method-sbert { background-color: #0a8754; }
    .method-hybrid { background-color: #b51d4a; }

    /* Table appearance tweak */
    .stTable table {
      border-collapse: separate;
      border-spacing: 0 8px;
    }
    .stTable tr { background: transparent; }
    .stTable td, .stTable th { padding: 8px 12px; vertical-align: middle; }

    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------
# CONFIG / DEFAULT PATHS (no sidebar)
# -----------------------------------------------------------
DATASET_OPTIONS = ["quora", "nfcorpus", "scifact", "fiqa", "arguana"]
# include hybrid variants with alpha values
METHOD_OPTIONS = ["BM25", "SBERT", "Hybrid (α=0.3)", "Hybrid (α=0.5)", "Hybrid (α=0.7)"]

PROJECT_RESULTS = Path(__file__).resolve().parents[1] / "results"
FALLBACK_RESULTS = Path("results")
ROOT_DEFAULT = PROJECT_RESULTS if PROJECT_RESULTS.exists() else FALLBACK_RESULTS
ROOT = ROOT_DEFAULT  # fixed; no UI for folder selection per request

# -----------------------------------------------------------
# Helper: discover result files and pick default for method
# -----------------------------------------------------------
def find_result_files(ds_dir: Path):
    files = {}
    if not ds_dir.exists():
        return files
    for p in sorted(ds_dir.glob("*.json")):
        key = p.name.replace(".json", "")
        files[key] = str(p)
    return files

def select_file_for_method(res_files, method):
    if not res_files:
        return None
    method_l = method.lower()
    for k, v in res_files.items():
        kl = k.lower()
        # treat any hybrid variant as hybrid
        if "hybrid" in method_l or "hybrid" in kl:
            if "hybrid" in kl or "alpha" in kl or "alpha" in k.lower() or "hybrid" in k.lower():
                return v
        if "bm25" in method_l and "bm25" in kl:
            return v
        if "sbert" in method_l and ("sbert" in kl or "mpnet" in kl or "sentence" in kl):
            return v
    # fallback: first file
    return list(res_files.values())[0]

# -----------------------------------------------------------
# BEIR loader (cached)
# -----------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_beir_data(dataset_name: str, root: Path):
    local_folder = Path("datasets") / dataset_name
    if local_folder.exists():
        try:
            split = "test" if dataset_name != "msmarco" else "dev"
            corpus, queries, qrels = GenericDataLoader(data_folder=str(local_folder)).load(split=split)
            corpus_texts = {cid: (v.get("title", "") + " " + v.get("text", "")).strip() for cid, v in corpus.items()}
            return corpus_texts, queries, qrels
        except Exception:
            pass
    # fallback: download once
    data_path = util.download_and_unzip(BASE_URL.format(dataset_name), "datasets")
    split = "test" if dataset_name != "msmarco" else "dev"
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    corpus_texts = {cid: (v.get("title", "") + " " + v.get("text", "")).strip() for cid, v in corpus.items()}
    return corpus_texts, queries, qrels

# -----------------------------------------------------------
# Ranked results loader (cached)
# -----------------------------------------------------------
@st.cache_data
def load_ranked_results(path):
    if path is None:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}
    out = {}
    for qid, val in raw.items():
        if isinstance(val, list):
            out[qid] = [(str(item[0]), float(item[1])) for item in val]
        elif isinstance(val, dict):
            out[qid] = sorted([(str(d), float(s)) for d, s in val.items()], key=lambda x: x[1], reverse=True)
        else:
            out[qid] = []
    return out

# -----------------------------------------------------------
# UI: Navbar + central controls (dataset, model, top-k)
# -----------------------------------------------------------
# centered navbar text only
st.markdown(
    '<div class="top-nav"><div style="font-size:20px; text-align:center; width:100%; font-weight:700;">Smart Document Semantic Search Engine</div></div>',
    unsafe_allow_html=True,
)

# options row: dataset, model (including hybrid variants), top-k in one horizontal line
opt_cols = st.columns([3, 3, 1.2])
with opt_cols[0]:
    dataset = st.selectbox("Dataset", DATASET_OPTIONS, index=0)
with opt_cols[1]:
    method_choice = st.selectbox("Model", METHOD_OPTIONS, index=0)
with opt_cols[2]:
    top_k = st.number_input("Top-K", min_value=1, max_value=200, value=10, step=1)

st.markdown("")  # small spacing

# -----------------------------------------------------------
# Load data and results based on selections (hidden path logic)
# -----------------------------------------------------------
ds_dir = ROOT / dataset
res_files = find_result_files(ds_dir)
selected_path = select_file_for_method(res_files, method_choice)
results_map = load_ranked_results(selected_path)

# Load BEIR dataset queries + corpus
with st.spinner("Loading queries and corpus (may download BEIR once)..."):
    corpus_texts, queries_dict, qrels = load_beir_data(dataset, ROOT)

query_ids = list(queries_dict.keys())
query_texts = [queries_dict[qid] for qid in query_ids]
if not query_texts:
    st.error("No query texts available for this dataset. Ensure BEIR dataset exists under 'datasets/<dataset>'.")
    st.stop()

# TF-IDF for matching free-text to saved BEIR queries
tfidf = TfidfVectorizer(stop_words="english", max_features=9000)
tfidf.fit(query_texts)
query_mat = tfidf.transform(query_texts)

# -----------------------------------------------------------
# Search bar (center)
# -----------------------------------------------------------
st.markdown('<div class="card"><h2 style="margin:0 0 6px 0;">Search</h2><div style="color:#5b6b8a">Enter a query to find the closest BEIR query and view precomputed results.</div></div>', unsafe_allow_html=True)
q = st.text_input("", value="", placeholder="Type your query and press Enter", key="main_search")

# -----------------------------------------------------------
# Matched BEIR query table and Results area (title change)
# -----------------------------------------------------------
if q:
    q_vec = tfidf.transform([q])
    sims = (query_mat @ q_vec.T).toarray().ravel()
    top_idx = np.argsort(-sims)[:1]
    matched_qids = [query_ids[i] for i in top_idx]
    matched_sims = [float(sims[i]) for i in top_idx]

    st.markdown('<div class="card"><h3 style="margin:0 0 8px 0;">Matched BEIR Query</h3></div>', unsafe_allow_html=True)
    df_matched = pd.DataFrame({
        "qid": matched_qids,
        "query_text": [queries_dict[qid] for qid in matched_qids],
        "similarity": matched_sims
    })
    st.table(df_matched)

    chosen = matched_qids[0]

    # Updated main results title as requested
    st.markdown(f'<div class="card"><h2 style="margin:0 0 6px 0;">Document Ranking Results (BM25 / SBERT / Hybrid) — Precomputed</h2></div>', unsafe_allow_html=True)

    if chosen not in results_map or not results_map.get(chosen):
        st.warning("No precomputed results available for the matched query.")
    else:
        hits = results_map[chosen][:top_k]
        # method badge: treat any 'hybrid' selection as hybrid class
        mc_low = method_choice.lower()
        if "bm25" in mc_low:
            badge_class = "method-bm25"
        elif "sbert" in mc_low:
            badge_class = "method-sbert"
        else:
            badge_class = "method-hybrid"

        st.markdown(f"<div style='margin-bottom:8px;'><span class='method-badge {badge_class}'>{method_choice}</span></div>", unsafe_allow_html=True)

        for i, (docid, score) in enumerate(hits, start=1):
            text = corpus_texts.get(str(docid), "(Document text not available)")
            snippet = (text[:600] + ("..." if len(text) > 600 else "")) if isinstance(text, str) else "(Document text not available)"
            st.markdown(
                f"""
                <div class="card">
                  <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="font-weight:700; font-size:15px;">{i}. Document: {docid}</div>
                    <div><span class="score-badge">{score:.4f}</span></div>
                  </div>
                  <div style="margin-top:10px; color:#3d4b63;">{snippet}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        perq_json = json.dumps({chosen: results_map[chosen]}, indent=2)
        st.download_button("Download this query's ranked list", data=perq_json, file_name=f"{dataset}_{chosen}_results.json")

else:
    st.info("Type a query above to see the matched BEIR query and precomputed ranking results.")

st.markdown("---")
# footer: updated as requested
st.markdown('<div style="text-align:center; color:#6b7280; padding-top:8px;">© 2025 Semantic Ranking System</div>', unsafe_allow_html=True)

#streamlit run "F:\B-Tech\project\New folder\app.py"