# Label-Aware Hierarchical Ranking Model (LHRM) for Multi-Label Text Classification

**Authors:** Lama Ayash, Abdulmohsen Algarni, and Omar Alqahtani  
**Affiliation:** Department of Computer Science, King Khalid University, Al Faraa, Abha, Aseer 61421, Saudi Arabia  
**Corresponding author:** lama.ayash@outlook.sa  
**Funding:** This work was supported by the Deanship of Scientific Research and Graduate Studies at King Khalid University under Research Grant **R.G.P.2/21/46**.

> **Abstract.** Multi-label text classification involves assigning multiple relevant categories to a single text, enabling applications in academic indexing, medical diagnostics, and e-commerce. However, existing models often fail to capture complex text–label relationships and lack robust mechanisms for ranking label relevance, limiting their effectiveness. This project presents the **Label-Aware Hierarchical Ranking Model (LHRM)**, which combines contextual embeddings, custom attention mechanisms, and gradient boosting to enhance label ranking. LHRM integrates document and label embeddings with a **Contextual Similarity Attention Module (CSAM)** to capture text–label relationships and inter-label dependencies, followed by a **Hierarchical Re-Ranking Module (HRRM)** for refining label prioritization. Experiments on **AAPD** and **Reuters-21578** show strong top-rank performance (e.g., **P@1: 86.90%** and **94.47%**, respectively), offering accurate, context-aware, and practically applicable multi-label predictions.

---

## ✨ Key Features
- **Label-enriched semantics:** Each label is represented with its full title + expanded definition to reduce ambiguity.
- **Contextual Similarity Attention (CSAM):** Aligns document and label embeddings; optional label–label self-attention.
- **Hierarchical Re-Ranking (HRRM):** Gradient Boosting post-processor that refines the order of predicted labels.
- **Ranking metrics:** Built-in **P@k** and **nDCG@k** for rigorous top-rank evaluation.
- **Simple CLI:** One-file training/evaluation and re-ranking pipeline (`lhrm.py`).

---

## 🏗️ Architecture (overview)
LHRM operates in **three stages**:
1. **Document & Label Embeddings:** BERT encodes input text and label definitions into contextual vectors.  
2. **CSAM:** Computes document–label similarity and (optionally) inter-label self-attention to refine label representations.  
3. **HRRM:** Trains a Gradient Boosting regressor on meta-features (initial score + rank) to adjust scores and re-rank labels.

> You can include your figures under `assets/` and reference them here, e.g.:
> - `assets/lhrm_architecture.png` (Fig. 2)
> - `assets/label_clusters.png` (Fig. 3)

---

## 📦 Installation
```bash
# (Optional) create a fresh environment
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate

pip install torch transformers scikit-learn pandas numpy tqdm
```

---

## 🚀 Quickstart

### 1) Train & Evaluate (save predictions for HRRM)
```bash
python lhrm.py train \
  --train_json /path/processed_train.json \
  --test_json  /path/processed_test.json \
  --labels_txt /path/labels.txt \
  --context_json /path/arxiv_labels_definitions_full.json \
  --variant csam_sa \
  --epochs 3 \
  --batch_size 16 \
  --bert_name bert-base-uncased \
  --max_len 128 \
  --out_preds /tmp/base_preds.json
```

### 2) Re-rank with HRRM (meta re-ranking)
```bash
python lhrm.py rerank \
  --preds_json /tmp/base_preds.json \
  --out_json   /tmp/reranked.json
```

**Supported variants:** `bert`, `bert+label`, `csam_nosa`, `csam_sa`

---

## 🧰 Data

- **AAPD** (ArXiv Academic Paper Dataset) – 55,840 abstracts, 54 labels.  
- **Reuters-21578** (ModApte split) – 9,788 articles, 90 labels (long tail).  

> **Note:** Do not commit raw datasets to the repo. Provide links and scripts to prepare `processed_train.json`, `processed_test.json`, and `labels.txt`. Label definitions file (e.g., `arxiv_labels_definitions_full.json`) should contain a mapping `{ "label": "short definition ...", ... }`.

### Expected JSON schema
- `processed_*.json` should be a list of objects with `"text"` and `"label"` (a string or list of strings):  
  ```json
  { "text": "document text ...", "label": ["cs.AI", "cs.LG"] }
  ```
- `labels.txt`: one label per line.  
- `context_json` (optional): `{ "cs.AI": "definition ...", "cs.LG": "definition ...", ... }`

---

## 📈 Evaluation
The script reports **Precision@k** and **nDCG@k** on the test split.  
LHRM is designed to excel at **top-rank** quality (P@1, nDCG@3/5).

Example metrics printed by `train` and `rerank`:
```
P@1: 0.8690
P@3: 0.6207
nDCG@3: 0.8211
nDCG@5: 0.8563
```

---

## 📊 Reproducibility Checklist
- Fixed random seed (`--seed 42` by default).
- Deterministic PyTorch backend where feasible.
- Clear preprocessing steps for datasets.
- Separate `train` and `test` JSONs with the same label space.
- Saved base predictions for HRRM (`--out_preds`), then re-rank via `rerank`.

---

## 🧪 Ablations (suggested)
- **Effect of CSAM:** `bert` vs `bert+label` vs `csam_nosa` vs `csam_sa`
- **Effect of HRRM:** before vs after re-ranking on the same base preds
- **Loss:** BCE vs Focal Loss
- **Label definitions:** with vs without `context_json`

> Add your actual numbers and figures (e.g., tables for AAPD/Reuters).

---

## 📜 Citation
If you find this work useful, please cite:

```bibtex
@article{Ayash2025LHRM,
  title   = {Label-Aware Hierarchical Ranking Model for Multi-Label Text Classification},
  author  = {Ayash, Lama and Algarni, Abdulmohsen and Alqahtani, Omar},
  journal = {IEEE Access},
  year    = {2025}
}
```

---

## 📄 License
- **Code:** Apache-2.0 (recommended; add `LICENSE` file).  
- **Paper text & figures (if included):** CC BY-NC-ND 4.0 as stated in the manuscript.

---

## 🙏 Acknowledgments
This work was supported by the Deanship of Scientific Research and Graduate Studies at King Khalid University under Research Grant **R.G.P.2/21/46**.
