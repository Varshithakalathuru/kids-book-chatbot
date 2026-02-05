import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.markdown("""
<style>
body {
    background-color: #FFF7E6;
}
h1, h2, h3 {
    color: #FF6F00;
}
</style>
""", unsafe_allow_html=True)

st.title("📚 Kids Book Chatbot 🤖")
st.write("Upload a book and ask questions from it 😊")

uploaded_file = st.file_uploader("📄 Upload a PDF book", type=["pdf"])

if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    st.success("✅ Book read successfully!")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)

    question = st.text_input("Ask me a question:")

    if question:
       docs = vectorstore.similarity_search(question, k=1)

       answer = docs[0].page_content

       st.write("🤖 Answer (Easy Explanation):")
       st.write("😊 Let me explain simply:")
       st.write(answer[:500])
