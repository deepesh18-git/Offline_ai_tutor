import os
import json


def chunk_text(text, chunk_size=400, overlap=50):

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text_str = ' '.join(chunk_words)

        if len(chunk_words) > 20:
            chunks.append(chunk_text_str)

        start = end - overlap

    return chunks


def save_chunks(chunks, filename, output_dir="data/chunks"):

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"{filename}_chunks.json"
    )

    data = {
        "filename": filename,
        "total_chunks": len(chunks),
        "chunks": chunks
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return output_path


def load_chunks(filename, chunks_dir="data/chunks"):

    file_path = os.path.join(
        chunks_dir, f"{filename}_chunks.json"
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Chunks file not found: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["chunks"]