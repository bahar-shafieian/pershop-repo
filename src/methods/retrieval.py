"""
BGE-m3 retrieval for RAG-ICL and the RAG-augmented merge.

k=2 neighbours used at training time for the merge; k=3 was empirically best for
frozen RAG-ICL (chosen on a 20-instance subset -- k-tests on small subsets are
noisy, see docs/04_engineering_lessons.md and docs/01_methodology.md 3.6.4).
"""
from typing import Optional

import numpy as np


class NeighbourRetriever:
    """Wraps a SentenceTransformer (BAAI/bge-m3) index over training instances
    of a single strategy, for nearest-neighbour retrieval by query similarity."""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.train_instances: list[dict] = []
        self.train_embeddings: Optional[np.ndarray] = None

    def build_index(self, train_instances: list[dict], batch_size: int = 64):
        self.train_instances = train_instances
        queries = [inst["query"] for inst in train_instances]
        self.train_embeddings = self.model.encode(
            queries, normalize_embeddings=True, batch_size=batch_size,
            show_progress_bar=True,
        )

    def retrieve(self, query: str, k: int = 2, exclude_id: Optional[str] = None) -> list[dict]:
        if self.train_embeddings is None:
            raise RuntimeError("Call build_index() first.")
        q_emb = self.model.encode([query], normalize_embeddings=True)
        scores = (self.train_embeddings @ q_emb.T).squeeze()
        ranked = np.argsort(-scores)
        neighbours = []
        for idx in ranked:
            inst = self.train_instances[idx]
            if exclude_id and inst.get("id") == exclude_id:
                continue
            neighbours.append(inst)
            if len(neighbours) == k:
                break
        return neighbours
