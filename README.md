# 🧠 Label-Aware Hierarchical Ranking Model (LHRM)  
### for Multi-Label Text Classification  

[![Paper](https://img.shields.io/badge/View%20on-IEEE%20Xplore-blue?logo=ieee&style=for-the-badge)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11129041)
[![License](https://img.shields.io/badge/License-Apache--2.0-green?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow?style=for-the-badge)](https://www.python.org/)
[![Torch](https://img.shields.io/badge/PyTorch-2.x-red?style=for-the-badge)](https://pytorch.org/)

---

**Authors:** Lama Ayash, Abdulmohsen Algarni, and Omar Alqahtani  
**Affiliation:** Department of Computer Science, King Khalid University, Al Faraa, Abha, Aseer 61421, Saudi Arabia  
**Corresponding author:** lama.ayash@outlook.sa  
**Funding:** Supported by the Deanship of Scientific Research and Graduate Studies at King Khalid University under Research Grant **R.G.P.2/21/46**.

> **Abstract.** Multi-label text classification involves assigning multiple relevant categories to a single text, enabling applications in academic indexing, medical diagnostics, and e-commerce. However, existing models often fail to capture complex text–label relationships and lack robust mechanisms for ranking label relevance. This project presents the **Label-Aware Hierarchical Ranking Model (LHRM)**, which combines contextual embeddings, custom attention mechanisms, and gradient boosting to enhance label ranking. LHRM integrates document and label embeddings with a **Contextual Similarity Attention Module (CSAM)** and a **Hierarchical Re-Ranking Module (HRRM)** for refining label prioritization. Experiments on **AAPD** and **Reuters-21578** show strong top-rank performance (e.g., **P@1: 86.90%** and **94.47%**), demonstrating accurate, context-aware, and practical multi-label predictions.

---

## ✨ Key Features
- 🔹 **Label-Enriched Semantics:** Each label uses its title + definition to reduce ambiguity.  
- 🔹 **Contextual Similarity Attention (CSAM):** Aligns document–label embeddings with optional label–label self-attention.  
- 🔹 **Hierarchical Re-Ranking (HRRM):** Gradient Boosting module refines predicted label order.  
- 🔹 **Ranking Metrics:** Built-in **P@k** and **nDCG@k** for rigorous top-rank evaluation.  
- 🔹 **Simple CLI:** One-file training, evaluation, and re-ranking pipeline (`lhrm.py`).  

---

## 🏗️ Architecture Overview

<p align="center">
  <img src="assets/LHRM.png" width="100%" alt="LHRM Architecture">
</p>

**Three main stages:**
1. **Document & Label Embeddings** — BERT encodes text and label definitions.  
2. **CSAM** — Computes text–label similarity + inter-label attention.  
3. **HRRM** — Gradient Boosting re-ranker adjusts label ordering.

---

## 📦 Installation
```bash
# (Optional) create a fresh environment
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate

pip install torch transformers scikit-learn pandas numpy tqdm
```

---

## 🚀 Quickstart

### 1️⃣ Train & Evaluate
```bash
python lhrm.py train   --train_json /path/processed_train.json   --test_json  /path/processed_test.json   --labels_txt /path/labels.txt   --context_json /path/arxiv_labels_definitions_full.json   --variant csam_sa   --epochs 3   --batch_size 16   --bert_name bert-base-uncased   --max_len 128   --out_preds /tmp/base_preds.json
```

### 2️⃣ Re-Rank with HRRM
```bash
python lhrm.py rerank   --preds_json /tmp/base_preds.json   --out_json   /tmp/reranked.json
```

**Supported variants:** `bert`, `bert+label`, `csam_nosa`, `csam_sa`

---

## 🧰 Data

- **AAPD** – 55,840 abstracts, 54 labels.  
- **Reuters-21578** – 9,788 news articles, 90 labels (long tail).  

> ⚠️ *Do not commit raw datasets.* Instead, include scripts or download links.  
> Label definitions (`context_json`) map each label to its short description.

**Example JSON:**
```json
{ "text": "document text ...", "label": ["cs.AI", "cs.LG"] }
```

---

## 📈 Evaluation

Reports **Precision@k** and **nDCG@k**:
```
P@1: 0.8690
P@3: 0.6207
nDCG@3: 0.8211
nDCG@5: 0.8563
```

LHRM focuses on *top-rank precision* (P@1, nDCG@3/5), critical for multi-label tasks.

---

## 🧪 Suggested Ablations
| Aspect | Variants / Notes |
|--------|------------------|
| CSAM | `bert` vs `bert+label` vs `csam_nosa` vs `csam_sa` |
| HRRM | before vs after re-ranking |
| Loss | BCE vs Focal Loss |
| Label Context | with vs without `context_json` |

---

## 📊 Reproducibility Checklist
- ✅ Fixed random seed (`--seed 42`)  
- ✅ Deterministic PyTorch backend (where feasible)  
- ✅ Separate `train` / `test` splits  
- ✅ Consistent label set  
- ✅ Full meta-feature re-ranking pipeline reproducibility  

---

## 📜 Citation
If you use LHRM in your research, please cite:
```bibtex
@article{Ayash2025LHRM,
  title   = {Label-Aware Hierarchical Ranking Model for Multi-Label Text Classification},
  author  = {Ayash, Lama and Algarni, Abdulmohsen and Alqahtani, Omar},
  journal = {IEEE Access},
  year    = {2025},
  doi     = {10.1109/ACCESS.2025.11129041}
}
```

---

## 📄 License
- Code: **Apache 2.0**  
- Figures & paper text: **CC BY-NC-ND 4.0**

---

## 🙏 Acknowledgments
Supported by the Deanship of Scientific Research and Graduate Studies at **King Khalid University** under Research Grant **R.G.P.2/21/46**.

---

<p align="center">
  <b>🔗 <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11129041">Read the Full Paper on IEEE Xplore</a></b>
</p>
