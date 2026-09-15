"""
Loads the trained TF-IDF vectorizer and models, and exposes prediction
functions used by the FastAPI app.
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from src.preprocessing import clean_text, combine_subject_body

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

DEFAULT_MODEL_NAME = "Linear SVM"  # best F1 score from training run

# ---------------------------------------------------------------------------
# Load artifacts once, at import time
# ---------------------------------------------------------------------------

with open(MODELS_DIR / "tfidf_vectorizer.pkl", "rb") as f:
    VECTORIZER = pickle.load(f)

with open(MODELS_DIR / "all_models.pkl", "rb") as f:
    _bundle = pickle.load(f)
    ALL_MODELS = _bundle["models"]          # dict: name -> fitted model
    TRAINING_METRICS = _bundle["results"]   # dict: name -> metrics dict

FEATURE_NAMES = np.array(VECTORIZER.get_feature_names_out())

# Precompute a "spam weight" per feature for each model, for explainability.
# - Naive Bayes: log P(word|spam) - log P(word|ham)
# - Logistic Regression / Linear SVM: model.coef_[0] (already spam-direction)
_SPAM_WEIGHTS = {}
for name, model in ALL_MODELS.items():
    if hasattr(model, "feature_log_prob_"):
        _SPAM_WEIGHTS[name] = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    elif hasattr(model, "coef_"):
        _SPAM_WEIGHTS[name] = model.coef_[0]
    else:
        _SPAM_WEIGHTS[name] = None


def available_models() -> list[str]:
    return list(ALL_MODELS.keys())


def get_model(name: Optional[str]):
    if name is None:
        name = DEFAULT_MODEL_NAME
    if name not in ALL_MODELS:
        raise ValueError(
            f"Unknown model '{name}'. Available: {', '.join(ALL_MODELS.keys())}"
        )
    return name, ALL_MODELS[name]


def _confidence(model, X) -> tuple[int, float]:
    """Return (predicted_label, confidence 0-1) for a single sample X."""
    pred = int(model.predict(X)[0])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = float(proba[pred])
    elif hasattr(model, "decision_function"):
        # LinearSVC: turn the signed distance into a pseudo-probability
        # via a logistic squash. Not a calibrated probability, but a
        # reasonable, monotonic confidence proxy.
        score = float(model.decision_function(X)[0])
        confidence = 1 / (1 + np.exp(-abs(score)))
    else:
        confidence = 1.0

    return pred, confidence


def _top_indicators(model_name: str, X, top_n: int = 8) -> list[dict]:
    """Words present in this email that most influenced the prediction,
    ranked by |tfidf_value * spam_weight|."""
    weights = _SPAM_WEIGHTS.get(model_name)
    if weights is None:
        return []

    row = X.toarray()[0]
    nonzero_idx = np.nonzero(row)[0]
    if len(nonzero_idx) == 0:
        return []

    contributions = row[nonzero_idx] * weights[nonzero_idx]
    order = np.argsort(np.abs(contributions))[::-1][:top_n]

    return [
        {
            "word": FEATURE_NAMES[nonzero_idx[i]],
            "direction": "spam" if contributions[i] > 0 else "not_spam",
            "weight": round(float(abs(contributions[i])), 4),
        }
        for i in order
    ]


def predict_email(subject: str, body: str, model_name: Optional[str] = None) -> dict:
    """Predict spam/not-spam for one email. Returns a JSON-serializable dict."""
    model_name, model = get_model(model_name)

    combined = combine_subject_body(subject, body)
    cleaned = clean_text(combined)
    X = VECTORIZER.transform([cleaned])

    pred, confidence = _confidence(model, X)
    indicators = _top_indicators(model_name, X)

    return {
        "label": "spam" if pred == 1 else "not_spam",
        "confidence": round(confidence, 4),
        "model_used": model_name,
        "top_indicators": indicators,
        "cleaned_text_preview": cleaned[:200],
    }


def predict_batch(texts: list[str], model_name: Optional[str] = None) -> list[dict]:
    """Predict for a list of raw email bodies (used by CSV batch upload)."""
    model_name, model = get_model(model_name)
    cleaned_texts = [clean_text(t) for t in texts]
    X = VECTORIZER.transform(cleaned_texts)

    preds = model.predict(X)
    if hasattr(model, "predict_proba"):
        confidences = model.predict_proba(X)
        confidences = [float(confidences[i][preds[i]]) for i in range(len(preds))]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        confidences = [float(1 / (1 + np.exp(-abs(s)))) for s in scores]
    else:
        confidences = [1.0] * len(preds)

    return [
        {
            "label": "spam" if p == 1 else "not_spam",
            "confidence": round(c, 4),
        }
        for p, c in zip(preds, confidences)
    ]


def model_metrics() -> dict:
    return TRAINING_METRICS
