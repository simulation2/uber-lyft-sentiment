# Cross-Brand Sentiment Classification of Uber vs Lyft Using Reddit Data and Machine Learning

> **Master’s Dissertation (M598, GISMA University of Applied Sciences)**  
> **Student:** Hossein Hami (GH1036065) · **Supervisor:** Ramin Baghaei Mehr  
> **Department:** Computer & Data Sciences (CDS) · **Year:** 2025

---

## 🔍 Overview

This repository contains the full, reproducible pipeline for **cross-brand sentiment classification** comparing **Uber** vs **Lyft** using **Reddit comments**.  
We build a clean NLP pipeline (cleaning → preprocessing → feature extraction → models → evaluation → analysis) and benchmark **classical ML** against a **BiLSTM** and a **blended ensemble**.

**Why it matters:** Understanding how riders discuss Uber vs Lyft helps product, ops, and marketing teams track **brand perception** and **experience gaps** across time and events.

---

## ✨ Key Contributions

- New Reddit corpus comparing Uber vs Lyft (balanced sampling, duplicates/NSFW removed).
- Consistent preprocessing and features: **TF–IDF (uni/bi-grams)** for classical ML; **embeddings** for BiLSTM.
- Benchmarks across **Complement Naïve Bayes, Logistic Regression, Linear SVM, Soft Voting, BiLSTM, Blended Ensemble**.
- Transparent evaluation (**accuracy, precision, recall, F1, confusion matrices**) and **error analysis**.
- Ethics-by-design: anonymization, paraphrasing of quotes, GDPR-conscious storage.

---

## 📦 Dataset

- **Source:** Reddit (public posts & comments).  
- **Brand filter:** `Uber`, `Lyft`.  
- **Example file (provided):** `reddit_comments_20250801_20250831__MERGED_ALL.csv` (Aug 01–31, 2025).  
- **Balanced sample (illustrative):** Uber **n ≈ 7,706**, Lyft **n ≈ 6,162**.  
- **Privacy:** No usernames retained. Quotes are paraphrased if specific/identifiable.

> If you cannot distribute raw data, keep a *data access note* (how to regenerate via the Reddit API / Pushshift / scraping helpers).

---

## 🧪 Methods (Pipeline)

1. **Data Collection** – Reddit API / Pushshift / scraping helpers.  
2. **Cleaning** – remove HTML, URLs, duplicates, emojis, and NSFW.  
3. **Preprocessing** – tokenize, lowercase, stopword removal, lemmatization.  
4. **Feature Extraction**  
   - **Classical ML:** TF–IDF (1–2 grams)  
   - **Deep Learning:** Embedding layer + BiLSTM (+ dropout)  
5. **Models**  
   - Complement Naïve Bayes (CNB)  
   - Logistic Regression (LR)  
   - Linear SVM (LinearSVC)  
   - Soft Voting Ensemble  
   - BiLSTM  
   - Blended Classical + LSTM Ensemble  
6. **Evaluation** – accuracy, precision, recall, macro-F1; confusion matrices.  
7. **Analysis** – ablation / sensitivity, class imbalance checks, error analysis.  
8. **Reporting** – figures & tables; reproducible code (+ GitHub).

A simple flowchart is in `figures/pipeline.png` (add your exported diagram there).

---

## 📈 Results (Summary)

| Model                         | Accuracy | Macro F1 | Notes |
|------------------------------|:--------:|:--------:|------|
| Complement Naïve Bayes (CNB) | **0.64** | ~0.62    | High recall on positive, weak on negatives |
| Logistic Regression           | **0.75** | ~0.73    | Strong baseline; robust & explainable |
| Linear SVM (LinearSVC)       | **0.75** | ~0.73    | Similar to LR; strong classical choice |
| Soft Voting (LR + SVM + CNB) | **0.74** | ~0.72    | Slightly below best single classical |
| BiLSTM                       | **0.77** | —        | Best single model (deep) |
| Blended Classical + LSTM     | **0.77** | —        | Ties BiLSTM; more stable in some folds |

> Add your full tables (per-class precision/recall/F1), confusion matrices, and learning curves in `results/`.

---

## 🛠️ Quickstart (Reproducible Setup)

### 1) Environment
```bash
# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate   # (Windows) .venv\Scripts\activate
pip install -r requirements.txt
```

Minimal `requirements.txt` (extend as needed):
```text
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
tensorflow==2.*
torch     # if you also test PyTorch-based baselines
tqdm
```

### 2) Data
Place CSV files under `data/`:
```
data/
  reddit_comments_20250801_20250831__MERGED_ALL.csv
  # or your regenerated exports
```

### 3) Run
This repository assumes a main pipeline script called **`thesis_accuracy_boost.py`**.

Common entry-points (adjust flags to your implementation):
```bash
# End-to-end (clean → preprocess → train → evaluate → export)
python thesis_accuracy_boost.py \
  --data_path data/reddit_comments_20250801_20250831__MERGED_ALL.csv \
  --models "cnb,lr,svm,softvote,bilstm,blend" \
  --test_size 0.15 --val_size 0.15 \
  --tfidf_ngrams 1 2 \
  --metrics "accuracy,precision,recall,f1" \
  --out_dir results/

# Train a specific model, e.g., LinearSVC
python thesis_accuracy_boost.py \
  --data_path data/reddit_comments_20250801_20250831__MERGED_ALL.csv \
  --models "svm" \
  --out_dir results/svm/
```

Outputs (expected):
```
results/
  metrics_overview.csv
  confusion_matrix_[MODEL].png
  classification_report_[MODEL].txt
  learning_curves_[MODEL].png
```

> Tip: For reproducibility, set random seeds and save a `run_config.json` with data splits and hyperparameters.

---

## 📂 Repository Structure (Suggested)

```
.
├─ data/                     # Raw or processed CSVs (do not commit raw if restricted)
├─ figures/                  # Flowchart, plots for README/thesis
├─ notebooks/                # EDA & experiments
├─ src/                      # Package-style modules (optional)
│  ├─ data.py                # loading, cleaning
│  ├─ preprocess.py          # tokenization, lemmatization, stopwords
│  ├─ features.py            # TF–IDF, embeddings
│  ├─ models.py              # CNB, LR, SVM, BiLSTM, ensemble
│  ├─ train.py               # training loops & CV
│  └─ eval.py                # metrics, plots
├─ results/                  # metrics, reports, confusion matrices
├─ thesis_accuracy_boost.py  # main pipeline runner
├─ requirements.txt
├─ LICENSE
└─ README.md
```

---

## 🔒 Ethics & Data Protection

- Public Reddit data only; **no usernames** stored.  
- Potentially identifiable quotes are **paraphrased**.  
- Data stored locally and not redistributed without platform terms & permissions.  
- Follows GISMA ethics guidance and GDPR principles (minimization, anonymization, purpose limitation).

For formal submission, keep your **Ethics Application & Consent** documentation in the thesis appendices (and link to it here if appropriate).

---

## 🧾 How to Cite

If you use this repository or dataset, please cite:

```bibtex
@mastersthesis{Hami2025UberLyftSentiment,
  title     = {Cross-Brand Sentiment Classification of Uber vs Lyft Using Reddit Data and Machine Learning},
  author    = {Hossein Hami},
  school    = {GISMA University of Applied Sciences},
  year      = {2025},
  address   = {Berlin, Germany},
  note      = {Supervisor: Ramin Baghaei Mehr}
}
```

Key background reading:
- Liu, B. (2012). *Sentiment Analysis and Opinion Mining*. Morgan & Claypool.  
- Pang, B., & Lee, L. (2008). *Opinion Mining and Sentiment Analysis*. Now Publishers.

(Extend references in your `thesis` and `docs/` folders.)

---

## 🪪 License

Specify your license (e.g., MIT, Apache-2.0). If data redistribution is restricted, **exclude raw data** from the repo and provide regeneration scripts instead.

---

## 📬 Contact

- **Hossein Hami** — reach out via GitHub or LinkedIn.  
- **Supervisor:** Ramin Baghaei Mehr.

---

### Notes for Reviewers/Examiners

- Code is modular and reproducible.  
- Results are accompanied by metrics and plots in `results/`.  
- Ethics and academic integrity considerations are documented (and reflected in how data are handled).

---

*Made with ❤️ for transparent, reproducible NLP research.*
