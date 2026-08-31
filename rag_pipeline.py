# app/rag_pipeline.py

import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from models.load_model import load_model, generate_response
from retriever.faiss_retriever import FAISSRetriever


class RAGPipeline:

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.retriever = None
        self.is_model_loaded = False
        self.is_index_loaded = False

    def load_language_model(self):
        self.tokenizer, self.model = load_model()
        self.is_model_loaded = True

    def load_retriever(self, index_name):
        self.retriever = FAISSRetriever()
        self.retriever.load_index(index_name)
        self.retriever.load_embedding_model()
        self.is_index_loaded = True

    def set_retriever(self, retriever):
        self.retriever = retriever
        self.is_index_loaded = True

    def build_prompt(self, context, question):
        return f"""Use the context below to answer the question.
If the answer is not in the context, say "I don't know based on the provided document."

Context:
{context}

Question: {question}

Answer:"""

    def answer(self, question, top_k=3, max_new_tokens=200):

        if not self.is_model_loaded:
            raise ValueError("Model not loaded.")

        if not self.is_index_loaded:
            raise ValueError("Retriever not loaded.")

        context = self.retriever.get_context(
            question, top_k=top_k
        )

        if not context:
            context = "No relevant content found."

        prompt = self.build_prompt(context, question)

        if len(prompt) > 2000:
            prompt = prompt[:2000]

        answer_text = generate_response(
            prompt,
            self.tokenizer,
            self.model,
            max_new_tokens=max_new_tokens
        )

        return {
            "question": question,
            "answer": answer_text,
            "context": context
        }


def process_pdf_to_index(pdf_path, index_name):

    from utils.pdf_reader import extract_text_from_pdf, save_raw_text
    from utils.text_cleaner import clean_text
    from utils.text_chunker import chunk_text
    from sentence_transformers import SentenceTransformer

    raw_text = extract_text_from_pdf(pdf_path)
    save_raw_text(raw_text, index_name)

    cleaned = clean_text(raw_text)
    chunks = chunk_text(cleaned, chunk_size=400, overlap=50)

    if not chunks:
        raise ValueError(
            "No chunks created. PDF may be empty or image-only."
        )

    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = emb_model.encode(
        chunks,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=16
    )

    retriever = FAISSRetriever()
    retriever.embedding_model = emb_model
    retriever.build_index(embeddings, chunks)
    retriever.save_index(index_name)

    return retriever