from __future__ import annotations

from typing import Optional

_MODEL = None


def get_model():
    """
    Process-local singleton SentenceTransformer.
    - Do NOT import/load heavy deps at import-time.
    - Load weights only on first call.
    """
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer("/root/models/minilm")
    return _MODEL