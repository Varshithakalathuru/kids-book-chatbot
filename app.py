import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="📚 Kids Book Chatbot",
    page_icon="📖",
    layout="wide"
)

# ---------------- TITLE ----------------
st.markdown("## 📘 Kids Book Chatbot")
st.markdown("Upload a kids story book PDF and ask questions in a fun way!")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Controls")
    pdf_file = st.file_uploader("Upload a PDF book", type="pdf")
    clear = st.button("🗑️ Clear Chat")

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if clear:
    st.session_state.messages = []
    st.session_state.vectorstore = None

# ---------------- FUNCTIONS ----------------
@st.cache_resource
def create_vectorstore(pdf):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


def get_qa_chain(vectorstore):
    llm = ChatOpenAI(
        temperature=0.3,
        model="gpt-3.5-turbo"
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff"
    )
    return qa

# ---------------- PDF PROCESSING ----------------
if pdf_file:
    with st.spinner("📖 Reading the book and getting smarter..."):
        vectorstore = create_vectorstore(pdf_file)
        st.session_state.vectorstore = vectorstore
    st.success("✅ Book loaded! Ask me anything 😊")

# ---------------- CHAT DISPLAY ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- USER INPUT ----------------
if st.session_state.get("vectorstore"):
    user_question = st.chat_input("Ask a question about the story...")

    if user_question:
        # show user message
        st.session_state.messages.append(
            {"role": "user", "content": user_question}
        )
        with st.chat_message("user"):
            st.markdown(user_question)

        # generate answer
        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                qa_chain = get_qa_chain(st.session_state.vectorstore)
                answer = qa_chain.run(
                    "Answer like a friendly assistant for kids. "
                    "Use simple words.\n\nQuestion: " + user_question
                )
                st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

else:
    st.info("👈 Upload a PDF from the sidebar to start chatting!")
