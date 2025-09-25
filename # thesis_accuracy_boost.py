# thesis_accuracy_boost.py
# Enhanced Uber vs Lyft Sentiment Analysis Pipeline with Classical + BiLSTM Blending


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # quiet TensorFlow INFO/WARN logs early

import re, random, warnings, numpy as np, pandas as pd, nltk, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================ SETTINGS ============================
BASE = "/Users/hosseinhami/Downloads"
DATA_BASENAME = "reddit_comments_20250801_20250831__MERGED_ALL"  # tries with/without .csv
OUT_DIR = os.path.join(BASE, "thesis_outputs")
RANDOM_STATE = 42
os.makedirs(OUT_DIR, exist_ok=True)

# ======================= DETERMINISTIC SEEDS ======================
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
try:
    import tensorflow as tf
    tf.random.set_seed(RANDOM_STATE)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
except Exception:
    pass

# ===================== QUIET NON-CRITICAL WARNS ====================
warnings.filterwarnings("ignore", category=UserWarning, module="keras")
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    pass

# ============================== NLTK ===============================
nltk.download("vader_lexicon", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

# ============================ HELPERS =============================
_CONTRACTIONS = {
    "can't":"can not","won't":"will not","n't":" not","'re":" are","'s":" is","'d":" would",
    "'ll":" will","'t":" not","'ve":" have","'m":" am"
}
def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"http\S+|www\S+"," URL ", s)
    s = re.sub(r"@\w+"," USER ", s)
    for k,v in _CONTRACTIONS.items(): s = s.replace(k, v)
    s = re.sub(r"([a-z])\1{2,}", r"\1\1", s)  # soooo -> soo
    s = re.sub(r"[^a-z\s]"," ", s)
    s = re.sub(r"\s+"," ", s).strip()
    return s

_stop = set(stopwords.words("english"))
for w in ("not","no","nor"): _stop.discard(w)
def strip_stop(s: str) -> str:
    toks = nltk.word_tokenize(s)
    return " ".join(t for t in toks if t not in _stop and len(t) > 1)

def _smooth(series: pd.Series, k=3) -> pd.Series:
    return series.rolling(k, min_periods=1).mean()

# ============================== LOAD ==============================
DATA_PATH = DATA_BASENAME if os.path.isabs(DATA_BASENAME) else os.path.join(BASE, DATA_BASENAME)
if not os.path.exists(DATA_PATH) and os.path.exists(DATA_PATH + ".csv"):
    DATA_PATH = DATA_PATH + ".csv"

df = pd.read_csv(DATA_PATH)

# ensure datetime
for cand in ("created_datetime","created_utc","created_at","created_time"):
    if cand in df.columns:
        df["created_datetime"] = pd.to_datetime(df[cand], errors="coerce", utc=True)
        break
if "created_datetime" not in df.columns:
    df["created_datetime"] = pd.to_datetime("now", utc=True)

TEXT_COL = "body" if "body" in df.columns else ("text" if "text" in df.columns else df.columns[0])
df[TEXT_COL] = df[TEXT_COL].astype(str)

# Optional balance (as previously used)
if "subreddit" in df.columns:
    df = df[~df["subreddit"].str.fullmatch(r"(?i)lyftdrivers", na=False)].copy()

# ======================= LABELS (VADER if missing) ==================
if "label" in df.columns and df["label"].nunique() in (2,3):
    if df["label"].dtype != int:
        mapping = {"negative":0,"neutral":1,"positive":2}
        df["label"] = df["label"].map(mapping).fillna(df["label"]).astype(int)
else:
    sia = SentimentIntensityAnalyzer()
    def vader_label_denoised(s: str) -> int:
        c = sia.polarity_scores(s)["compound"]
        if c >= 0.15: return 2
        if c <= -0.15: return 0
        return 1
    df["label"] = df[TEXT_COL].apply(vader_label_denoised).astype(int)

# ============================= CLEANING ============================
df["clean_text"] = df[TEXT_COL].map(normalize_text).map(strip_stop)
df = df[df["clean_text"].str.len() > 0].copy()

# ============================= SPLIT ==============================
from sklearn.model_selection import train_test_split
y = df["label"].values
X_text = df["clean_text"].values

X_train_text, X_temp_text, y_train, y_temp = train_test_split(
    X_text, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)
X_val_text, X_test_text, y_val, y_test = train_test_split(
    X_temp_text, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
)

# ============== TF-IDF (WORDS + CHARS) + CHI² SELECT ===============
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.feature_selection import SelectKBest, chi2

word_tfidf = TfidfVectorizer(
    min_df=5, max_df=0.95, ngram_range=(1,2), analyzer="word",
    sublinear_tf=True, smooth_idf=True, strip_accents="unicode"
)
char_tfidf = TfidfVectorizer(
    min_df=3, ngram_range=(3,6), analyzer="char_wb",
    sublinear_tf=True, smooth_idf=True
)

Xw_tr = word_tfidf.fit_transform(X_train_text)
Xc_tr = char_tfidf.fit_transform(X_train_text)
X_train_raw = hstack([Xw_tr, Xc_tr], format="csr")

Xw_va = word_tfidf.transform(X_val_text)
Xc_va = char_tfidf.transform(X_val_text)
X_val_raw = hstack([Xw_va, Xc_va], format="csr")

Xw_te = word_tfidf.transform(X_test_text)
Xc_te = char_tfidf.transform(X_test_text)
X_test_raw = hstack([Xw_te, Xc_te], format="csr")

K = min(150000, max(10000, X_train_raw.shape[1] * 3 // 4))
selector = SelectKBest(score_func=chi2, k=K).fit(X_train_raw, y_train)
X_train = selector.transform(X_train_raw)
X_val   = selector.transform(X_val_raw)
X_test  = selector.transform(X_test_raw)

# ===================== MODELS + PARAM SEARCH =======================
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, GridSearchCV

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# NB
nb_params = {"alpha":[0.2, 0.5, 0.8, 1.0]}
nb_gs = GridSearchCV(ComplementNB(), nb_params, scoring="f1_macro", cv=cv5, n_jobs=-1, verbose=0)
nb_gs.fit(X_train, y_train); nb_best = nb_gs.best_estimator_

# LR (add gentle neutral emphasis option)
lr_params = {
    "C":[0.5,1.0,2.0,3.0],
    "class_weight":[{0:1.0,1:1.2,2:1.0}, {0:1.0,1:1.4,2:1.0}, "balanced"],
    "penalty":["l2"],
    "solver":["lbfgs","liblinear"],
    "max_iter":[2000]
}
lr_gs = GridSearchCV(LogisticRegression(), lr_params, scoring="f1_macro", cv=cv5, n_jobs=-1, verbose=0)
lr_gs.fit(X_train, y_train); lr_best = lr_gs.best_estimator_

# LinearSVC (calibrated) — correct param names via 'estimator__'
svm_base = LinearSVC(class_weight="balanced")
svm_cal  = CalibratedClassifierCV(estimator=svm_base, cv=3, method="isotonic")
svm_params = {"estimator__C":[0.5,0.8,1.0,2.0]}
svm_gs = GridSearchCV(svm_cal, svm_params, scoring="f1_macro", cv=cv5, n_jobs=-1, verbose=0)
svm_gs.fit(X_train, y_train); svm_best = svm_gs.best_estimator_

# Fit on train
nb_best.fit(X_train, y_train)
lr_best.fit(X_train, y_train)
svm_best.fit(X_train, y_train)

# ============== VALIDATION-WEIGHTED SOFT VOTE (CLASSICAL) ==========
from sklearn.metrics import f1_score, log_loss

def proba(model, X): return model.predict_proba(X)
y_nb_val  = nb_best.predict(X_val);  f_nb  = f1_score(y_val, y_nb_val, average="macro")
y_lr_val  = lr_best.predict(X_val);  f_lr  = f1_score(y_val, y_lr_val, average="macro")
y_svm_val = svm_best.predict(X_val); f_svm = f1_score(y_val, y_svm_val, average="macro")
w_sum = max(1e-6, f_nb + f_lr + f_svm)
w_nb, w_lr, w_svm = f_nb/w_sum, f_lr/w_sum, f_svm/w_sum

def soft_vote_proba(X):
    return w_nb*proba(nb_best, X) + w_lr*proba(lr_best, X) + w_svm*proba(svm_best, X)

def soft_vote_predict(X): return soft_vote_proba(X).argmax(axis=1)

# ============================== BiLSTM ==============================
try:
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, GlobalMaxPool1D, SpatialDropout1D
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
except Exception:
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.models import Sequential
    from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, GlobalMaxPool1D, SpatialDropout1D
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from keras.optimizers import Adam

from sklearn.utils.class_weight import compute_class_weight
MAX_WORDS = 30000
MAX_LEN   = 160
EMB_DIM   = 128

tok = Tokenizer(num_words=MAX_WORDS, lower=True, oov_token="<unk>")
tok.fit_on_texts(X_train_text)
def to_pad(texts): return pad_sequences(tok.texts_to_sequences(texts), maxlen=MAX_LEN, padding="post", truncating="post")
Xtr_pad = to_pad(X_train_text); Xva_pad = to_pad(X_val_text); Xte_pad = to_pad(X_test_text)

cls = np.unique(y_train)
raw_w = compute_class_weight(class_weight="balanced", classes=cls, y=y_train)
class_weight = {int(c): float(w) for c, w in zip(cls, raw_w)}

vocab_size = min(MAX_WORDS, len(tok.word_index)+1)
lstm = Sequential([
    Embedding(vocab_size, EMB_DIM),
    SpatialDropout1D(0.2),
    Bidirectional(LSTM(128, return_sequences=True, dropout=0.25, recurrent_dropout=0.25)),
    GlobalMaxPool1D(),
    Dropout(0.35),
    Dense(128, activation="relu"),
    Dropout(0.25),
    Dense(3, activation="softmax"),
])
lstm.compile(optimizer=Adam(learning_rate=2e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
es  = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1)
lstm.fit(Xtr_pad, y_train, validation_data=(Xva_pad, y_val), epochs=18, batch_size=64,
         callbacks=[es, rlr], class_weight=class_weight, verbose=2)

# ================== LSTM TEMPERATURE SCALING (VAL-BASED) ===========
def _softmax_temperature(logits, T):
    eps = 1e-8
    p = np.asarray(logits)
    # If probabilities supplied, convert to logits first
    if p.ndim == 2 and np.allclose(p.sum(axis=1), 1, atol=1e-3):
        p = np.clip(p, eps, 1 - eps)
        p = np.log(p)
    z = p / float(T)
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)

def fit_temperature_on_val(p_val, y_val):
    best_T, best_ll = 1.0, 1e9
    for T in np.linspace(0.6, 2.0, 15):
        pT = _softmax_temperature(p_val, T)
        ll = log_loss(y_val, pT, labels=[0,1,2])
        if ll < best_ll:
            best_ll, best_T = ll, T
    return best_T

# Raw LSTM probs
p_lstm_val_raw  = lstm.predict(Xva_pad, verbose=0)
p_lstm_test_raw = lstm.predict(Xte_pad, verbose=0)
# Fit temperature on validation, apply to val/test
T_lstm = fit_temperature_on_val(p_lstm_val_raw, y_val)
p_lstm_val  = _softmax_temperature(p_lstm_val_raw,  T_lstm)
p_lstm_test = _softmax_temperature(p_lstm_test_raw, T_lstm)
y_lstm      = p_lstm_test.argmax(axis=1)

# ============================ EVALUATION ===========================
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
LABELS = ["negative","neutral","positive"]

def _to_labels(y_pred):
    y_pred = np.asarray(y_pred)
    return y_pred.argmax(axis=1) if y_pred.ndim == 2 else y_pred

def evaluate_model(name, y_true, y_pred):
    y_pred = _to_labels(y_pred)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n{name} Classification Report:")
    print(classification_report(y_true, y_pred, target_names=LABELS, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4.8,4.3))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{name} - Confusion Matrix")
    plt.xticks(range(3), LABELS, rotation=45); plt.yticks(range(3), LABELS)
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(os.path.join(OUT_DIR, f"{name.replace(' ','_').lower()}_cm.png"), dpi=170)
    plt.close()
    return {"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}

def save_report(name, y_true, y_pred, path):
    y_pred = _to_labels(y_pred)
    with open(path, "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=LABELS, zero_division=0))

# Classical preds
y_nb   = nb_best.predict(X_test)
y_lr   = lr_best.predict(X_test)
y_svm  = svm_best.predict(X_test)
y_soft = soft_vote_predict(X_test)

# Blend classical proba with calibrated LSTM using validation log-loss
p_class_val  = soft_vote_proba(X_val)
p_class_test = soft_vote_proba(X_test)

alphas = np.linspace(0.1, 0.9, 17)
best_alpha, best_ll = 0.5, 1e9
for a in alphas:
    p_mix = a * p_class_val + (1 - a) * p_lstm_val
    ll = log_loss(y_val, p_mix, labels=[0,1,2])
    if ll < best_ll:
        best_alpha, best_ll = a, ll

p_mix_test = best_alpha * p_class_test + (1 - best_alpha) * p_lstm_test
y_mix      = p_mix_test.argmax(axis=1)

# Metrics & plots
perf = []
perf.append(evaluate_model("ComplementNB (tuned+chi2)", y_test, y_nb))
perf.append(evaluate_model("LogReg (tuned+chi2)",       y_test, y_lr))
perf.append(evaluate_model("LinearSVC (tuned+chi2)",    y_test, y_svm))
perf.append(evaluate_model("SoftVote NB+LR+SVM (val-wt)", y_test, y_soft))
perf.append(evaluate_model("BiLSTM (calibrated)",       y_test, y_lstm))
perf.append(evaluate_model(f"Blended Classical+LSTM (α={best_alpha:.2f})", y_test, y_mix))

# Save text reports
save_report("NB",     y_test, y_nb,   os.path.join(OUT_DIR, "report_nb.txt"))
save_report("LR",     y_test, y_lr,   os.path.join(OUT_DIR, "report_lr.txt"))
save_report("SVM",    y_test, y_svm,  os.path.join(OUT_DIR, "report_svm.txt"))
save_report("SoftEn", y_test, y_soft, os.path.join(OUT_DIR, "report_softvote.txt"))
save_report("LSTM",   y_test, y_lstm, os.path.join(OUT_DIR, "report_lstm.txt"))
save_report("Blend",  y_test, y_mix,  os.path.join(OUT_DIR, "report_blend.txt"))

perf_df = pd.DataFrame(perf).set_index("Model")
perf_df.to_csv(os.path.join(OUT_DIR, "performance_table.csv"))
plt.figure(figsize=(7.6,4.2))
perf_df[["Accuracy","Precision","Recall","F1"]].plot(kind="bar", rot=0)
plt.ylim(0,1.0); plt.title("Model Performance Comparison (Final)")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "model_performance.png"), dpi=170); plt.close()

# ===================== ASPECT ANALYSIS (COMBINED) ===================
PRICING_PATTERNS = [
    r"\bprice\b", r"\bpricing\b", r"\bfare(?:s)?\b", r"\bcost(?:s)?\b", r"\brate(?:s)?\b",
    r"\bsurge\b", r"\bsurge pricing\b", r"\bexpensive\b", r"\bcheap(?:er)?\b"
]
DRIVER_PATTERNS = [
    r"\bdriver(?:s)?\b", r"\buber driver\b", r"\blyft driver\b", r"\bcab driver\b",
    r"\brude\b", r"\bpolite\b", r"\bfriendly\b", r"\bunfriendly\b",
    r"\bprofessional\b", r"\bunprofessional\b", r"\battitude\b", r"\bbehaviou?r\b"
]

def aspect_slice(frame: pd.DataFrame, patterns):
    pat = "|".join(patterns)
    m = frame[TEXT_COL].str.contains(pat, case=False, regex=True, na=False)
    return frame.loc[m].copy()

df["sentiment"] = df["label"].map({0:"negative",1:"neutral",2:"positive"})
df["date"] = pd.to_datetime(df["created_datetime"], errors="coerce", utc=True).dt.date

# Brand (Uber/Lyft)
if "brand" not in df.columns:
    if "subreddit" in df.columns:
        df["brand"] = np.where(df["subreddit"].str.contains("lyft", case=False, na=False), "Lyft", "Uber")
    else:
        df["brand"] = np.where(df[TEXT_COL].str.contains(r"\blyft\b", case=False, na=False), "Lyft",
                        np.where(df[TEXT_COL].str.contains(r"\buber\b", case=False, na=False), "Uber", np.nan))
    df = df[df["brand"].isin(["Uber","Lyft"])].copy()

pricing_df = aspect_slice(df, PRICING_PATTERNS)
driver_df  = aspect_slice(df, DRIVER_PATTERNS)

def combined_brand_aspect_trend(aspect_df: pd.DataFrame, aspect_name: str, out_name: str, smooth_k=3):
    if aspect_df.empty: return
    tmp = (aspect_df.groupby(["date","brand"])["sentiment"]
                    .value_counts(normalize=True)
                    .rename("prop")
                    .reset_index())
    tmp = tmp[tmp["sentiment"].isin(["positive","neutral","negative"])]
    trend = tmp.pivot_table(index="date", columns=["brand","sentiment"], values="prop", fill_value=0.0)

    # Ensure all columns exist
    cols = [("Uber","positive"),("Uber","neutral"),("Uber","negative"),
            ("Lyft","positive"),("Lyft","neutral"),("Lyft","negative")]
    for c in cols:
        if c not in trend.columns: trend[c] = 0.0
    trend = trend.reindex(columns=cols).sort_index()

    # Optional smoothing
    for c in trend.columns:
        trend[c] = _smooth(trend[c], k=smooth_k)

    # Consistent brand colors; sentiment via line style
    BRAND_COLOR = {"Uber":"#1f77b4", "Lyft":"#d62728"}   # blue, red
    SENT_STYLE  = {"positive":"solid", "neutral":"dashed", "negative":"dashdot"}

    plt.figure(figsize=(12.8,5.2))
    for brand in ["Uber","Lyft"]:
        for sent in ["positive","neutral","negative"]:
            s = trend[(brand, sent)]
            plt.plot(trend.index, s,
                     label=f"{brand} {sent}",
                     color=BRAND_COLOR[brand],
                     linestyle=SENT_STYLE[sent],
                     linewidth=2.2)
    plt.title(f"{aspect_name} Sentiment Trend Over Time — Uber vs Lyft", fontsize=20, pad=10)
    plt.xlabel("Date", fontsize=14); plt.ylabel("Proportion", fontsize=14)
    plt.xticks(rotation=35); plt.legend(ncol=2, fontsize=11)
    plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig(os.path.join(OUT_DIR, out_name), dpi=170); plt.close()
    trend.to_csv(os.path.join(OUT_DIR, out_name.replace(".png",".csv")))

combined_brand_aspect_trend(pricing_df, "Pricing", "pricing_uber_vs_lyft_combined.png", smooth_k=3)
combined_brand_aspect_trend(driver_df,  "Driver Behavior", "driver_behavior_uber_vs_lyft_combined.png", smooth_k=3)

print("\nSaved figures and metrics to:", OUT_DIR)
