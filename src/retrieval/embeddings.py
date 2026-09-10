import pickle
from pathlib import Path
from typing import List, Optional
import numpy as np


class EmbeddingEncoder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_st: bool = False):
        self.model_name = model_name
        self.model = None
        self.use_st = use_st
        self.fallback_vectorizer = None

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 384))

        if self.use_st and self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
            except Exception:
                self.model = None

        if self.model is not None:
            return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        # Fast normalized TF-IDF embedding (384 dimensions)
        from sklearn.feature_extraction.text import TfidfVectorizer
        if self.fallback_vectorizer is None:
            self.fallback_vectorizer = TfidfVectorizer(max_features=384, token_pattern=r"(?u)\b\w+\b")
            matrix = self.fallback_vectorizer.fit_transform(texts).toarray()
        else:
            matrix = self.fallback_vectorizer.transform(texts).toarray()

        if matrix.shape[1] < 384:
            padded = np.zeros((matrix.shape[0], 384))
            padded[:, :matrix.shape[1]] = matrix
            matrix = padded

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def save_vectorizer(self, filepath: str):
        if self.fallback_vectorizer is not None:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(self.fallback_vectorizer, f)

    def load_vectorizer(self, filepath: str):
        path = Path(filepath)
        if path.exists():
            with open(path, "rb") as f:
                self.fallback_vectorizer = pickle.load(f)
