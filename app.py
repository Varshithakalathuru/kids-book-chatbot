import streamlit as st
from pypdf import PdfReader

st.title("📚 Kids Book Chatbot 🤖")
st.write("Upload a book and ask questions from it 😊")

uploaded_file = st.file_uploader("📄 Upload a PDF book", type=["pdf"])

if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        text += page.extract_text()

    st.success("✅ Book read successfully!")
    st.text_area("📖 Book Content (preview)", text[:2000])
