# Offline AI Educational Tutor

An offline RAG-based educational chatbot that answers questions 
from uploaded PDF documents using Small Language Models.

## Architecture
PDF → Text Extraction → Chunking → Embeddings → FAISS → SLM → Answer

## Tech Stack
- TinyLlama / Phi-2 (Language Model)
- Sentence Transformers (Embeddings)
- FAISS (Vector Search)
- Streamlit (UI)
- PyPDF (PDF Processing)

## Setup

git clone <your-repo>
cd offline_ai_tutor
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

## Run

streamlit run app.py

## Usage
1. Click Load Model in sidebar
2. Upload educational PDF
3. Click Process PDF
4. Ask questions in chat
