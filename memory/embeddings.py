"""Lightweight embedding utilities with deterministic local vectors."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass


_SYNONYM_MAP = {
    "报错": "error",
    "错误": "error",
    "故障": "error",
    "异常": "error",
    "网关": "gateway",
    "网关层": "gateway",
    "超时": "timeout",
    "502": "gateway error",
    "bad gateway": "gateway error",
}


@dataclass(slots=True)
class EmbeddingConfig:
    dimension: int = 128


class LocalEmbeddingEngine:
    """Deterministic local embedding implementation with keyword normalization."""

    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig()
        self.dimension = max(16, int(self.config.dimension))

    def embed(self, text: str) -> list[float]:
        """Generate a normalized dense vector from text."""
        raw_text = (text or "").strip()
        if not raw_text:
            return [0.0] * self.dimension
        if "[[force_embedding_error]]" in raw_text:
            raise ValueError("forced embedding error")

        normalized = self._normalize_text(raw_text)
        tokens = self._tokenize(normalized)
        if not tokens:
            return [0.0] * self.dimension

        vec = [0.0] * self.dimension
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = 1.0 + (digest[5] / 255.0) * 0.5
            vec[index] += sign * weight

        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]

    @staticmethod
    def cosine_similarity(left: list[float], right: list[float]) -> float:
        """Compute cosine similarity, robust to zero vectors."""
        if not left or not right:
            return 0.0
        if len(left) != len(right):
            return 0.0
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return float(numerator / (left_norm * right_norm))

    def _normalize_text(self, text: str) -> str:
        lowered = text.lower()
        for key, replacement in _SYNONYM_MAP.items():
            lowered = lowered.replace(key, f" {replacement} ")
        lowered = re.sub(r"[^\w\u4e00-\u9fff+#.-]+", " ", lowered)
        return re.sub(r"\s+", " ", lowered).strip()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        ascii_tokens = re.findall(r"[a-z0-9_+#.-]{2,}", text)
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
        # Merge individual Chinese chars into bi-grams for slightly richer semantics.
        chinese_bigrams = [
            chinese_chars[i] + chinese_chars[i + 1]
            for i in range(len(chinese_chars) - 1)
        ]
        return [*ascii_tokens, *chinese_bigrams]
