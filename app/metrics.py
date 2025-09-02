from __future__ import annotations
import math
import re
from typing import List
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from sklearn.feature_extraction.text import HashingVectorizer
except Exception:
    HashingVectorizer = None

_EMB = None
_VEC = None
_DISCOURSE_MARKERS = [
    r"\bfirst(ly)?\b", r"\bsecond(ly)?\b", r"\bthird(ly)?\b",
    r"\bnext\b", r"\bthen\b", r"\bfinally\b", r"\bin conclusion\b",
    r"\boverview\b", r"\bbackground\b", r"\bmethods?\b", r"\bresults?\b", r"\bdiscussion\b",
]

def _ensure_models():
    """Lazy init an embedding model or a cheap fallback."""
    global _EMB, __VEC
    if _EMB is None and SentenceTransformer is not None:
        try:
            _EMB = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _EMB = None
    if _EMB is None and _VECTOR_FALLBACK_AVAILABLE():
        if _VecNotInit():
            _init_vec()

def _VECTOR_FALLBACK_AVAILABLE():
    return HashingVectorizer is not None

def _VecNotInit():
    return (_V := globals().get("_VEC")) is None

def _init_vec():
    globals()["_VEC"] = HashingVectorizer(n_features=768, alternate_sign=False, norm=None)

def _encode(texts: List[str]) -> np.ndarray:
    _ensure_models()
    if _EMB is not None:
        embs = _EMB.encode(texts, normalize_embeddings=True)
        return np.asarray(embs, dtype=np.float32)
    X = _VEC.transform(texts) if hasattr(_VEC, "transform") else _VEC.fit_transform(texts)
    X = X.astype(np.float32)
    norms = np.sqrt((X.multiply(X)).sum(axis=1)).A1 + 1e-8
    X = X.multiply(1.0 / norms[:, None])
    return X.toarray()

def split_paragraphs(text: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"\n{2,}|\r{2,}", text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def structure_score(text: str) -> float:
    paras = split_paragraphs(text)
    if not paras:
        return 0.0
    heading_hits = sum(bool(re.match(r"^(\d+\.|\#|\*\*)\s", p)) for p in paras)
    cue_hits = sum(bool(re.search("|".join(_DISCOURSE_MARKERS), p, flags=re.I)) for p in paras)
    surface = min(1.0, (heading_hits + 0.5 * cue_hits) / max(3, len(paras)))
    emb = _encode(paras)
    sims = [float(np.dot(emb[i], emb[i + 1])) for i in range(len(paras) - 1)] or [0.0]
    cohesion = max(0.0, min(1.0, float(np.mean(sims))))
    words = max(1, len(re.findall(r"\w+", text)))
    trans = 200.0 * sum(bool(re.search(p, text, flags=re.I)) for p in _DISCOURSE_MARKERS) / words
    transitions = max(0.0, min(1.0, trans))
    return round(0.4 * surface + 0.3 * cohesion + 0.3 * transitions, 4)

def plan_adherence(outline: List[str], text: str) -> float:
    if not outline:
        return 0.0
    paras = split_paragraphs(text)
    titles = [re.sub(r"[\W_]+", " ", t.lower()).strip() for t in outline]
    hits = []
    for t in titles:
        idxs = [i for i, p in enumerate(paras) if t and t in re.sub(r"[\W_]+", " ", p.lower())]
        hits.append(min(idxs) if idxs else None)

    coverage = sum(h is not None for h in hits) / len(titles)
    order_idxs = [h for h in hits if h is not None]
    if len(order_idxs) < 2:
        order = 0.5
    else:
        inv = sum(order_idxs[i] > order_idxs[j] for i in range(len(order_idxs)) for j in range(i + 1, len(order_idxs)))
        denom = len(order_idxs) * (len(order_idxs) - 1) / 2
        order = 1.0 - inv / denom if denom else 1.0
    return round(0.6 * coverage + 0.4 * order, 4)

def topic_drift(outline: List[str], text: str) -> float:
    paras = split_paragraphs(text)
    if not paras:
        return 1.0
    base_text = " ".join(outline) if outline else " ".join(paras[:2])
    par = _encode(paras)
    base = _encode([base_text])[0]
    one_minus_cos = [max(0.0, min(1.0, 1.0 - float(np.dot(v, base)))) for v in par]
    return round(float(np.mean(one_minus_cos)), 4)

def hallucination_proxy(text: str) -> float:
    numerals = len(re.findall(r"\b\d{2,}\b", text))
    proper = len(re.findall(r"\b[A-Z][a-z]{2,}\b(?:\s[A-Z][a-z]{1,}\b)?", text))
    links = len(re.findall(r"https?://", text)) + len(re.findall(r"\[\d+\]", text))
    suspicious = max(0, (numerals + int(0.3 * proper) - links))
    length_k = max(1, len(text) // 1000)
    return round(min(1.0, suspicious / (3 * length_k)), 4)

__all__ = [
    "structure_score",
    "plan_adherence",
    "topic_drift",
    "hallucination_proxy",
]