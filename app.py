import streamlit as st
from pypdf import PdfReader

st.title("📄 PDF Test App")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    reader = PdfReader(uploaded_file)
    st.success("PDF loaded successfully ✅")
    st.write("Number of pages:", len(reader.pages))
