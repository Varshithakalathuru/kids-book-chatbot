import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

st.title("📚 Kids Book Chatbot 🤖")

uploaded_file = st.file_uploader("📄 Upload a PDF book", type=["pdf"])

if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        text += page.extract_text()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)

    st.success("Book is ready! Ask me anything 😊")

    question = st.text_input("Ask a question:")

    if question:
        docs = vectorstore.similarity_search(question, k=1)
        st.write("🤖 Answer:")
        st.write(docs[0].page_content)
