# app.py

import streamlit as st
import os
import sys
import tempfile

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Offline AI Educational Tutor",
    page_icon="📚",
    layout="wide"
)

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

if "index_ready" not in st.session_state:
    st.session_state.index_ready = False


@st.cache_resource
def load_pipeline():
    from app.rag_pipeline import RAGPipeline
    pipeline = RAGPipeline()
    pipeline.load_language_model()
    return pipeline


def process_uploaded_pdf(uploaded_file):
    from app.rag_pipeline import process_pdf_to_index
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf"
        ) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        index_name = os.path.splitext(uploaded_file.name)[0]
        index_name = "".join(
            c if c.isalnum() else "_"
            for c in index_name
        )

        retriever = process_pdf_to_index(tmp_path, index_name)
        st.session_state.pipeline.set_retriever(retriever)
        st.session_state.index_ready = True
        os.unlink(tmp_path)
        return True

    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return False


# ── UI Layout ──────────────────────────────────────

st.title("📚 Offline AI Educational Tutor")
st.divider()

# ── Sidebar ────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Setup")

    # Load Model
    st.subheader("1. Load Model")
    if not st.session_state.model_loaded:
        if st.button(
            "🚀 Load Model",
            type="primary",
            use_container_width=True
        ):
            with st.spinner("Loading model..."):
                try:
                    st.session_state.pipeline = load_pipeline()
                    st.session_state.model_loaded = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.success("✅ Model Ready")

    st.divider()

    # Upload PDF
    st.subheader("2. Upload PDF")
    if st.session_state.model_loaded:
        uploaded_file = st.file_uploader(
            "Choose PDF", type=["pdf"]
        )
        if uploaded_file:
            if st.button(
                "📥 Process PDF",
                type="primary",
                use_container_width=True
            ):
                with st.spinner("Processing PDF..."):
                    success = process_uploaded_pdf(
                        uploaded_file
                    )
                if success:
                    st.success("✅ Ready!")
                    st.rerun()
    else:
        st.info("Load model first.")

    st.divider()

    # Settings
    st.subheader("3. Settings")
    top_k = st.slider("Context chunks", 1, 5, 3)
    max_tokens = st.slider("Max response length", 100, 500, 200)

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    # Status
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Model",
            "✅" if st.session_state.model_loaded else "❌"
        )
    with col2:
        st.metric(
            "Index",
            "✅" if st.session_state.index_ready else "❌"
        )


# ── Main Chat ──────────────────────────────────────

chat_col, context_col = st.columns([2, 1])

with chat_col:
    st.subheader("💬 Chat")

    for question, answer in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)

    if not st.session_state.chat_history:
        st.info("Upload a PDF and ask a question.")

    st.divider()

    if (st.session_state.model_loaded
            and st.session_state.index_ready):

        with st.form("chat_form", clear_on_submit=True):
            user_question = st.text_input(
                "Question",
                placeholder="Ask something from the PDF...",
                label_visibility="collapsed"
            )
            submitted = st.form_submit_button(
                "Send 📤",
                type="primary",
                use_container_width=True
            )

        if submitted and user_question.strip():
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.pipeline.answer(
                        question=user_question,
                        top_k=top_k,
                        max_new_tokens=max_tokens
                    )
                    st.session_state.chat_history.append(
                        (user_question, result["answer"])
                    )
                    st.session_state.last_context = (
                        result["context"]
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        if not st.session_state.model_loaded:
            st.warning("Load the model from sidebar.")
        elif not st.session_state.index_ready:
            st.warning("Upload and process a PDF first.")


with context_col:
    st.subheader("🔍 Retrieved Context")

    if hasattr(st.session_state, "last_context"):
        with st.expander("View Source Chunks", expanded=False):
            st.text(st.session_state.last_context)
    else:
        st.info("Context will appear here after first question.")