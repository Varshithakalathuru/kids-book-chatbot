import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

# --------- Page Styling ----------
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

# --------- User Personalization ----------
age = st.slider("Select your age:", 5, 15)
favorite_genre = st.selectbox("Choose your favorite genre:", ["Adventure", "Mystery", "Science", "Fantasy", "Animals"])

# --------- Upload PDF ----------
uploaded_file = st.file_uploader("📄 Upload a PDF book", type=["pdf"])

if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    st.success("✅ Book read successfully!")

    # --------- Split text into chunks ----------
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    # --------- Create embeddings ----------
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # --------- Question Input ----------
    question = st.text_input("Ask me a question:")

    if question:
        # Search top 3 similar chunks
        docs = vectorstore.similarity_search(question, k=3)
        answer = " ".join([doc.page_content for doc in docs])

        # Optional: Summarize the answer
        summarizer = pipeline("summarization")
        short_answer = summarizer(answer, max_length=200, min_length=50, do_sample=False)[0]['summary_text']

        # Display
        st.write(f"🤖 Answer for a {age}-year-old who likes {favorite_genre}:")
        st.write(short_answer)
