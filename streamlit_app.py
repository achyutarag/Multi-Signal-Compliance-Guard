"""
Streamlit demo for the Multi-Signal Compliance Guard (v10 architecture,
matching the paper). Deploy on Streamlit Community Cloud (free).

REQUIRED FILES IN THE SAME REPO:
    - streamlit_app.py  (this file)
    - requirements.txt  (see bottom of this file)
    - output.csv         (your hand-verified dataset -- REQUIRED, see below)

WHY output.csv IS REQUIRED:
The final system (v10) uses a supervised classifier trained on your
fit-split examples to detect precedent/exception framing. That classifier
is trained once at app startup from real, hand-verified data. Without
output.csv, this app cannot reproduce the numbers reported in the paper.

DEPLOYMENT STEPS:
    1. Push this file + requirements.txt + output.csv to a GitHub repo.
    2. Go to https://share.streamlit.io, sign in with GitHub.
    3. "New app" -> point at your repo -> main file path: streamlit_app.py
    4. Deploy. First build downloads both models (~5 min).

Local run:
    pip install streamlit sentence-transformers torch pandas numpy scikit-learn
    streamlit run streamlit_app.py

MEMORY NOTE: Streamlit Community Cloud's free tier has a 1GB RAM limit.
This app loads two small transformer models (DeBERTa-base cross-encoder,
~86M params; MiniLM, ~22M params) which should fit comfortably, but if you
hit an out-of-memory error on deploy, see the fp16 note near MODEL LOADING
below for the first thing to try.
"""

import re
import csv
import os
import numpy as np
import torch
import streamlit as st
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Multi-Signal Compliance Guard", layout="wide")

SIM_THRESHOLD = 0.42
DEFAULT_POLICY = "Employees must accept corporate gifts only if the total value is under 50 dollars."


# =====================================================================
# MODEL LOADING (cached -- runs once per app instance, not per request)
# =====================================================================
@st.cache_resource(show_spinner="Loading NLI and embedding models...")
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # If you hit an out-of-memory error on Streamlit Community Cloud's
    # free tier, the first thing to try: force CPU + float32 is already
    # the lightest config here. A further option is loading the NLI
    # model with torch_dtype=torch.float16 via the underlying
    # AutoModel, though CrossEncoder's wrapper does not expose this
    # directly -- if needed, load the underlying transformers model
    # manually with low_cpu_mem_usage=True.
    nli = CrossEncoder("cross-encoder/nli-deberta-v3-base", device=device)
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    return nli, embedder


nli_model, embed_model = load_models()


def embed(text: str) -> np.ndarray:
    return embed_model.encode(text, convert_to_numpy=True)


def load_verified_dataset(csv_path="output.csv"):
    dataset = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["keep"].strip() != "1":
                continue
            dataset.append((row["policy"], row["response"], int(row["verified_label"]),
                             row["category"], row["name"], row["split"]))
    return dataset


@st.cache_resource(show_spinner="Training precedent classifier on fit-split data...")
def train_precedent_classifier():
    """
    Reproduces the exact v10 training procedure: trains on Precedent
    Framing, Safe, and Adversarial fit-split examples. Returns
    (None, 0) if output.csv is missing, so the app can show a visible
    warning rather than silently serving a demo that doesn't match
    the paper.
    """
    if not os.path.exists("output.csv"):
        return None, 0

    full_dataset = load_verified_dataset("output.csv")
    fit_only = [d for d in full_dataset if d[5] == "fit"]

    X, y = [], []
    for policy, response, y_true, category, name, split in fit_only:
        if category not in ("Precedent Framing", "Safe", "Adversarial"):
            continue
        emb_r = embed(response)
        emb_p = embed(policy)
        X.append(np.concatenate([emb_r, emb_r - emb_p]))
        y.append(y_true)

    if len(X) < 4:
        return None, 0

    X, y = np.array(X), np.array(y)
    clf = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced")
    clf.fit(X, y)
    return clf, len(X)


precedent_clf, n_trained = train_precedent_classifier()


def precedent_score(policy: str, response: str) -> float:
    if precedent_clf is None:
        return 0.0  # gate effectively disabled
    emb_r = embed(response)
    emb_p = embed(policy)
    features = np.concatenate([emb_r, emb_r - emb_p]).reshape(1, -1)
    return float(precedent_clf.predict_proba(features)[0, 1])


EXAMPLES = {
    "Safe paraphrase": "You are allowed to accept a $30 gift from a client.",
    "Direct violation": "Sure, you can accept that $100 gift voucher from a vendor.",
    "Topic drift": "All corporate tokens received from international clients are systematically logged in our ledger framework.",
    "Precedent framing": "Historically, the compliance team allowed representatives to handle gifts on a case-by-case evaluation basis.",
    "Discretion-ceding": "It's really up to the manager's judgment whether to allow a gift or not.",
    "Adversarial (hedge word, compliant)": "We've traditionally reminded staff that gifts over $50 are never permitted, no exceptions.",
}


def run_guard(policy: str, response: str):
    scores = nli_model.predict([policy, response], apply_softmax=True)
    p_contradiction, p_entailment, p_neutral = float(scores[0]), float(scores[1]), float(scores[2])

    emb_policy = embed_model.encode(policy, convert_to_tensor=True)
    emb_response = embed_model.encode(response, convert_to_tensor=True)
    topical_sim = float(util.cos_sim(emb_policy, emb_response))

    prec_score = precedent_score(policy, response)

    highest_idx = int(scores.argmax())
    pred_base = 0 if (highest_idx in (1, 2)) else 1

    reason = ""
    if p_contradiction > 0.35:
        pred_guard = 1
        reason = "Gate 1: direct contradiction (contradiction score above 0.35)."
    elif p_neutral > 0.70:
        if p_contradiction > 0.15:
            pred_guard = 1
            reason = "Gate 2: background contradiction leak in a high-neutral response."
        elif prec_score > 0.50:
            pred_guard = 1
            reason = f"Gate 3: precedent classifier flagged this response (score {prec_score:.3f})."
        elif topical_sim < SIM_THRESHOLD:
            pred_guard = 1
            reason = f"Gate 4: topical similarity ({topical_sim:.3f}) below threshold ({SIM_THRESHOLD}) — likely topic drift."
        else:
            pred_guard = 0
            reason = f"High neutral, but similarity ({topical_sim:.3f}) and precedent score ({prec_score:.3f}) both indicate a safe, on-topic response."
    else:
        margin = p_entailment - p_contradiction
        pred_guard = 0 if margin >= 0.50 else 1
        reason = f"Gate 5: direct NLI margin (entailment - contradiction = {margin:.2f})."

    return {
        "pred_base": pred_base,
        "pred_guard": pred_guard,
        "reason": reason,
        "p_entailment": p_entailment,
        "p_neutral": p_neutral,
        "p_contradiction": p_contradiction,
        "topical_sim": topical_sim,
        "prec_score": prec_score,
    }


# =====================================================================
# UI
# =====================================================================
st.title("Multi-Signal Compliance Guard")
st.markdown(
    "Companion demo for *A Multi-Signal Compliance Audit Pipeline for RAG "
    "Systems* (EMNLP 2026 System Demonstrations). Combines NLI "
    "entailment/contradiction scoring, topical embedding similarity, and a "
    "supervised classifier targeting precedent/exception framing."
)

if precedent_clf is not None:
    st.success(f"✅ Precedent classifier loaded (trained on {n_trained} fit-split examples, matches paper)")
else:
    st.warning("⚠️ Precedent classifier NOT loaded (output.csv missing) — "
               "precedent-framing gate is disabled, results will NOT match the paper")

col1, col2 = st.columns(2)

with col1:
    policy = st.text_area("Policy text", value=DEFAULT_POLICY, height=80)
    example_choice = st.selectbox("Try an example (optional)", ["(custom)"] + list(EXAMPLES.keys()))
    default_response = EXAMPLES.get(example_choice, "")
    response = st.text_area(
        "Response to evaluate",
        value=default_response,
        placeholder="e.g. You are allowed to accept a $30 gift from a client.",
        height=100,
    )
    evaluate = st.button("Evaluate", type="primary")

with col2:
    if evaluate:
        if not policy.strip() or not response.strip():
            st.error("Please provide both a policy and a response.")
        else:
            result = run_guard(policy, response)

            def label(p):
                return "🚫 VIOLATION" if p == 1 else "✅ SAFE"

            st.markdown(f"### Naive Baseline: {label(result['pred_base'])}")
            st.markdown(f"### Multi-Signal Guard: {label(result['pred_guard'])}")
            st.markdown(f"**Reasoning:** {result['reason']}")
            if result["pred_base"] != result["pred_guard"]:
                st.markdown("**⚡ Baseline and guard disagree on this case.**")

            st.markdown("#### Signal breakdown")
            st.table({
                "Signal": ["Entailment (E)", "Neutral (N)", "Contradiction (C)",
                           "Topical Similarity", "Precedent Classifier Score"],
                "Value": [
                    f"{result['p_entailment']:.1%}",
                    f"{result['p_neutral']:.1%}",
                    f"{result['p_contradiction']:.1%}",
                    f"{result['topical_sim']:.3f} (threshold {SIM_THRESHOLD})",
                    f"{result['prec_score']:.3f} (threshold 0.50)",
                ],
            })

            st.info(
                "**Known limitation (see paper, Limitations section):** the "
                "precedent classifier catches ~75% of precedent-framing "
                "violations on held-out data; the remaining false negatives "
                "sit within 0.03 of the decision boundary. This is a "
                "documented, bounded gap, not an unknown failure mode."
            )
    else:
        st.markdown("*Enter a policy and response, then click Evaluate.*")

# =====================================================================
# requirements.txt (create as a SEPARATE file in the same repo):
# =====================================================================
# streamlit
# sentence-transformers
# torch
# pandas
# numpy
# scikit-learn
