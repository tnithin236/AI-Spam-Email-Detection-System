"""
Text preprocessing for SpamShield.

IMPORTANT: This must match the cleaning applied when the TF-IDF vectorizer
and models were trained (see the training notebook). If you change this,
your predictions will be inaccurate because the vectorizer was fit on text
cleaned this exact way.
"""

import re
import string

import nltk
from nltk.corpus import stopwords

# Ensure stopwords are available (no-op if already downloaded)
try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOP_WORDS = set(stopwords.words("english"))

_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
_URL_RE = re.compile(r"http\S+|www\S+")
_EMAIL_RE = re.compile(r"\S+@\S+")
_DIGIT_RE = re.compile(r"\d+")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/emails/punctuation/numbers, remove stopwords."""
    text = str(text).lower()
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text)
    text = _DIGIT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()

    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 1]
    return " ".join(tokens)


def combine_subject_body(subject: str, body: str) -> str:
    """Combine subject + body into one text blob before cleaning.

    Subject is repeated once more since spam signal is often concentrated
    there (e.g. "WINNER!!! Claim your prize") and TF-IDF benefits from the
    slightly higher term frequency.
    """
    subject = subject or ""
    body = body or ""
    return f"{subject} {subject} {body}"
