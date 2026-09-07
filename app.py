# app.py

import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

from agent import agent, extract_text

load_dotenv()

st.title("AI Research Assistant")
st.write("Ask questions about your document, do math, or search the web.")

# ------------------------------------------------
# File upload + ingestion
# ------------------------------------------------
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    # only re-ingest if this is a NEW file (avoid re-processing on every rerun)
    if st.session_state.get("last_uploaded") != uploaded_file.name:
        with st.spinner("Processing document..."):
            # Step 1: save the uploaded file to a temporary location on disk
            # (PyPDFLoader needs an actual file path, not the raw upload object)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            # Step 2: load the PDF
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()

            # Step 3: split into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)

            # Step 4: create embeddings
            embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

            # Step 5: store in Chroma (overwriting old data by using a fresh directory)
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )

            # Step 6: clean up the temp file
            os.remove(tmp_path)

            # remember this file so we don't re-process it every time the page reruns
            st.session_state["last_uploaded"] = uploaded_file.name

        st.success(f"Processed and indexed: {uploaded_file.name} ({len(chunks)} chunks)")
    else:
        st.info(f"Using already-indexed file: {uploaded_file.name}")

# ------------------------------------------------
# Chat (unchanged from before)
# ------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_question = st.chat_input("Ask a question...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = agent.invoke({"messages": [{"role": "user", "content": user_question}]})
                answer = extract_text(result["messages"][-1].content)
            except Exception as e:
                answer = f"Error: {e}"
            st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})