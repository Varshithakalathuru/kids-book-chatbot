import streamlit as st
from PyPDF2 import PdfReader

st.title("PDF Test App")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    reader = PdfReader(uploaded_file)
    st.write("Number of pages:", len(reader.pages))
