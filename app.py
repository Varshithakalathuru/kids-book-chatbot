import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="📚 Kids Book Chatbot",
    page_icon="📖",
    layout="wide"
)

st.title("📘 Kids Book Chatbot (100% Free)")
st.write("Upload a kids story book PDF and ask questions 😊")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Controls")
    pdf_file = st.file_uploader("Upload a PDF book", type="pdf")
    clear = st.button("🗑️ Clear Chat")

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if clear:
    st.session_state.messages = []
    st.session_state.index = None
    st.session_state.chunks = []

# ---------------- MODELS ----------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    qa = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_length=150,
        do_sample=False
    )

    return embedder, qa

# ---------------- PDF PROCESSING ----------------
def process_pdf(pdf):
    reader = PdfReader(pdf)
    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    if not text.strip():
        return None, None

    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    embedder, _ = load_models()
    embeddings = embedder.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return index, chunks

# ---------------- QA ----------------
def answer_question(question):
    embedder, qa = load_models()

    q_embedding = embedder.encode([question])
    D, I = st.session_state.index.search(np.array(q_embedding), k=1)
    context = st.session_state.chunks[I[0][0]]

    prompt = f"""
Answer the question in simple English for a child.

Context:
{context}

Question:
{question}

Answer:
"""

    result = qa(prompt)
    answer = result[0]["generated_text"]

    return answer   # 🔥 THIS WAS MISSING

# ---------------- LOAD PDF ----------------
if pdf_file and st.session_state.index is None:
    with st.spinner("📖 Reading the book..."):
        index, chunks = process_pdf(pdf_file)

    if index is None:
        st.error("❌ This PDF has no readable text.")
    else:
        st.session_state.index = index
        st.session_state.chunks = chunks
        st.success("✅ Book loaded! Ask me anything 🎉")

# ---------------- CHAT ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.index:
    question = st.chat_input("Ask a question about the story...")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                answer = answer_question(question)
                st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
else:
    st.info("👈 Upload a PDF from the sidebar to start chatting")
