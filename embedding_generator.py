# embeddings/embedding_generator.py

import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_embedding_model():
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model


def generate_embeddings(chunks, model):
    embeddings = model.encode(
        chunks,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )
    return embeddings


def save_embeddings(embeddings, chunks, filename,
                    output_dir="embeddings"):

    os.makedirs(output_dir, exist_ok=True)

    emb_path = os.path.join(
        output_dir, f"{filename}_embeddings.npy"
    )
    np.save(emb_path, embeddings)

    chunks_path = os.path.join(
        output_dir, f"{filename}_chunks.pkl"
    )
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    return emb_path, chunks_path


def load_embeddings(filename, embeddings_dir="embeddings"):

    emb_path = os.path.join(
        embeddings_dir, f"{filename}_embeddings.npy"
    )
    chunks_path = os.path.join(
        embeddings_dir, f"{filename}_chunks.pkl"
    )

    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f"Embeddings not found: {emb_path}"
        )

    embeddings = np.load(emb_path)

    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    return embeddings, chunks