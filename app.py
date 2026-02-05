import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.markdown(
    """
    <style>
    body {
        background-color:  #FFF0F5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📚 Kids Book Chatbot 🤖")
st.write("Upload a book and ask questions from it 😊")
st.markdown("### 🤖 Hi! I’m Book Buddy!")
st.write("📖 I read your book and answer your questions in a fun way!")
st.write("✨ Ask me anything from your story!")


uploaded_file = st.file_uploader("📄 Upload a PDF book", type=["pdf"])

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Question input (OUTSIDE condition ✅)
question = st.text_input("Ask me a question:")

if uploaded_file:
    text = ""
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        text += page.extract_text()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings()
    st.session_state.vectorstore = FAISS.from_texts(chunks, embeddings)

    st.success("✅ Book loaded! Now ask questions 👆")

# Answer section
if question and st.session_state.vectorstore:
    docs = st.session_state.vectorstore.similarity_search(question, k=1)
    st.markdown("### 🧠 Here’s what I found:")
    st.write(docs[0].page_content)
    st.write("😊 Want to ask another question?")
    st.write("🌟 You’re doing great! Keep asking questions!")


