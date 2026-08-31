# retriever/faiss_retriever.py

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class FAISSRetriever:

    def __init__(self):
        self.index = None
        self.chunks = []
        self.embedding_model = None
        self.embedding_dim = 384

    def load_embedding_model(self):
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL
        )

    def build_index(self, embeddings, chunks):
        self.chunks = chunks
        self.embedding_dim = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings.astype(np.float32))

    def save_index(self, filename, save_dir="embeddings"):
        os.makedirs(save_dir, exist_ok=True)

        index_path = os.path.join(
            save_dir, f"{filename}_faiss.bin"
        )
        faiss.write_index(self.index, index_path)

        chunks_path = os.path.join(
            save_dir, f"{filename}_retriever_chunks.pkl"
        )
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)

    def load_index(self, filename, save_dir="embeddings"):
        index_path = os.path.join(
            save_dir, f"{filename}_faiss.bin"
        )
        chunks_path = os.path.join(
            save_dir, f"{filename}_retriever_chunks.pkl"
        )

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        self.index = faiss.read_index(index_path)

        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

    def retrieve(self, query, top_k=3):
        if self.index is None:
            raise ValueError("Index not built or loaded.")

        if self.embedding_model is None:
            self.load_embedding_model()

        query_embedding = self.embedding_model.encode([query])
        query_embedding = query_embedding.astype(np.float32)

        distances, indices = self.index.search(
            query_embedding, top_k
        )

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(dist)))

        return results

    def get_context(self, query, top_k=3):
        results = self.retrieve(query, top_k)

        context_parts = [
            f"[Source {i+1}]\n{chunk}"
            for i, (chunk, _) in enumerate(results)
        ]

        return "\n\n".join(context_parts)