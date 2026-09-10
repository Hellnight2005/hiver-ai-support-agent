import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from pydantic import BaseModel, Field
from src.retrieval.embeddings import EmbeddingEncoder


class RetrievalItem(BaseModel):
    conversation_id: str
    customer_message: str
    agent_response: str
    intent: Optional[str] = "unknown_other"
    brand: str


class VectorIndex:
    def __init__(self, encoder: Optional[EmbeddingEncoder] = None):
        self.encoder = encoder or EmbeddingEncoder()
        self.items: List[RetrievalItem] = []
        self.vectors: Optional[np.ndarray] = None

    def build_index(self, items: List[RetrievalItem]):
        self.items = items
        if not items:
            self.vectors = np.empty((0, 384))
            return

        texts = [it.customer_message for it in items]
        self.vectors = self.encoder.encode(texts)

    def search(
        self,
        query: str,
        top_k: int = 5,
        exclude_conv_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        if not self.items or self.vectors is None or len(self.vectors) == 0:
            return []

        exclude_set = set(exclude_conv_ids or [])
        q_vec = self.encoder.encode([query])[0]

        # Cosine similarity
        norms = np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(q_vec)
        norms[norms == 0] = 1e-10
        scores = np.dot(self.vectors, q_vec) / norms

        sorted_indices = np.argsort(scores)[::-1]
        results = []

        for idx in sorted_indices:
            item = self.items[idx]
            if item.conversation_id in exclude_set:
                continue

            results.append({
                "conversation_id": item.conversation_id,
                "customer_message": item.customer_message,
                "agent_response": item.agent_response,
                "intent": item.intent,
                "similarity": float(scores[idx])
            })

            if len(results) >= top_k:
                break

        return results

    def save(self, directory: str = "artifacts/retrieval_index"):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        items_data = [it.model_dump() for it in self.items]
        with open(path / "items.json", "w", encoding="utf-8") as f:
            json.dump(items_data, f, indent=2)

        if self.vectors is not None:
            np.save(path / "vectors.npy", self.vectors)

        self.encoder.save_vectorizer(str(path / "vectorizer.pkl"))

    @classmethod
    def load(cls, directory: str = "artifacts/retrieval_index", encoder: Optional[EmbeddingEncoder] = None) -> "VectorIndex":
        path = Path(directory)
        index = cls(encoder=encoder)
        
        if (path / "vectorizer.pkl").exists():
            index.encoder.load_vectorizer(str(path / "vectorizer.pkl"))

        if (path / "items.json").exists():
            with open(path / "items.json", "r", encoding="utf-8") as f:
                items_data = json.load(f)
            index.items = [RetrievalItem(**d) for d in items_data]

        if (path / "vectors.npy").exists():
            index.vectors = np.load(path / "vectors.npy")

        return index
