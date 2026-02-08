import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="📚 Kids Book Chatbot",
    page_icon="📖",
    layout="wide"
)

st.title("📘 Kids Book Chatbot")
st.write("Upload a kids story book PDF and ask questions 😊")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Controls")
    pdf_file = st.file_uploader("Upload a PDF book", type="pdf")
    clear = st.button("🗑️ Clear Chat")

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if clear:
    st.session_state.messages = []
    st.session_state.vectorstore = None

# ---------------- FUNCTIONS ----------------
@st.cache_resource
def create_vectorstore(pdf):
    reader = PdfReader(pdf)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    if not text.strip():
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    if not chunks:
        return None

    embeddings = OpenAIEmbeddings()  # ✅ SAFE ON STREAMLIT CLOUD

    return FAISS.from_texts(chunks, embeddings)


def get_chain(vectorstore):
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template(
        """
        You are a friendly assistant for kids.
        Answer in simple words and short sentences.

        Context:
        {context}

        Question:
        {input}
        """
    )

    doc_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    return create_retrieval_chain(retriever, doc_chain)

# ---------------- PDF PROCESSING ----------------
if pdf_file and st.session_state.vectorstore is None:
    with st.spinner("📖 Reading the book..."):
        vectorstore = create_vectorstore(pdf_file)

    if vectorstore is None:
        st.error("❌ This PDF has no readable text. Please upload a text-based PDF.")
    else:
        st.session_state.vectorstore = vectorstore
        st.success("✅ Book loaded! Ask me anything 🎉")

# ---------------- CHAT HISTORY ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- USER INPUT ----------------
if st.session_state.vectorstore:
    question = st.chat_input("Ask a question about the story...")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                try:
                    chain = get_chain(st.session_state.vectorstore)
                    result = chain.invoke({"input": question})
                    answer = result["answer"]
                except Exception as e:
                    answer = "⚠️ Something went wrong. Please try again."

                st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
else:
    st.info("👈 Upload a PDF from the sidebar to start chatting")
