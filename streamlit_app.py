import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import tempfile
import os

# Sidebar UI
st.sidebar.title("⚖️ AI Paralegal Assistant")
openai_api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")

# Main UI
st.title("📄 Upload a Legal Document (PDF only)")
uploaded_file = st.file_uploader("", type=["pdf"])

if uploaded_file and openai_api_key:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.success("✅ File uploaded!")

    # Load and split document
    loader = PyMuPDFLoader(tmp_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    # Embed and store in Chroma
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = Chroma.from_documents(split_docs, embedding=embeddings)

    # Setup QA
    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    # Question input
    st.header("🧠 Ask About This Document")
    query = st.text_input("Your question:")

    if query:
        with st.spinner("Thinking..."):
            answer = qa.run(query)
        st.subheader("Answer:")
        st.write(answer)

    os.remove(tmp_path)
elif uploaded_file:
    st.warning("Please enter your OpenAI key in the sidebar.")
