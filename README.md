# Hybrid Approach to Smart Document Search Using Semantic Ranking (BM25 + SBERT)

This repository presents a hybrid information retrieval system that combines **BM25-based lexical search** with **SBERT-based semantic retrieval**, enhanced using **FAISS** for efficient and scalable similarity search. The project bridges traditional keyword-based retrieval and modern embedding-based search to improve relevance across diverse query types.

---

## 📌 Project Overview

Traditional keyword-based search systems often fail to capture semantic intent, while purely semantic models may overlook exact keyword relevance. This project proposes a **hybrid ranking framework** that fuses both approaches to deliver robust and accurate document retrieval.

The system is evaluated on multiple datasets from the **BEIR benchmark**, demonstrating consistent improvements in top-K relevance metrics across domains such as scientific, biomedical, financial, and argumentative text.

---

## 🧠 Key Features

- Hybrid ranking using **BM25 + SBERT**
- **FAISS** for fast nearest-neighbor search on dense embeddings
- Evaluation on multiple **BEIR benchmark datasets**
- Support for different fusion weights (α = 0.3, 0.5, 0.7)
- Interactive **Streamlit-based web interface**
- Precomputed rankings for low-latency retrieval

---

## 🔧 Technologies & Tools

- BM25
- Sentence-BERT (SBERT)
- FAISS
- Python
- NLP
- BEIR Benchmark
- Streamlit
- Scikit-learn
- NumPy, Pandas

---

## 🏗️ System Architecture

1. **Query Understanding**
   - User query matched to closest BEIR query using TF-IDF and cosine similarity
    <img width="1041" height="708" alt="Screenshot 2025-12-23 102856" src="https://github.com/user-attachments/assets/1ff96648-a324-492d-b861-1b6b35daee1e" />



2. **Document Ranking**
   - Lexical ranking using BM25
   - Semantic ranking using SBERT embeddings + FAISS
   - Hybrid score fusion using weighted normalization
    

3. **Result Presentation**
   - Top-K ranked documents with relevance scores and text snippets
   - Downloadable ranked results in JSON format
   <img width="704" height="812" alt="Screenshot 2025-12-23 102920" src="https://github.com/user-attachments/assets/6821499a-ca6c-4e79-ae09-67d0584d21e1" />

---

## 📊 Evaluation

The system is evaluated using standard Information Retrieval metrics:
- Precision@K
- Recall@K
- nDCG@K
- MAP

Experiments show that the hybrid approach consistently outperforms individual BM25 and SBERT models across multiple datasets.
![WhatsApp Image 2025-12-23 at 10 33 46 AM](https://github.com/user-attachments/assets/57570851-43ea-4a3f-83e6-2c00abe58e6a)

---

## 🚀 Running the Application

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt

