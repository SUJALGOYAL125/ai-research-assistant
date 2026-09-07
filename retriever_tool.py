# retriever_tool.py

import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.tools import tool

def get_api_key(key_name):
    try:
        return st.secrets[key_name]
    except Exception:
        return os.environ.get(key_name)

# make sure GOOGLE_API_KEY is available in the environment either way
os.environ["GOOGLE_API_KEY"] = get_api_key("GOOGLE_API_KEY")

# Step 1: set up the embeddings (same as in rag_query.py)
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# Step 2: load your existing Chroma database (already filled from Day 1)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Step 3: create a retriever that fetches the top 2 matching chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


# Step 4: wrap it as a tool the agent can call
@tool
def pdf_retriever(question: str) -> str:
    """Search the uploaded PDF document to answer questions about its content."""
    docs = retriever.invoke(question)          # get the matching chunks
    if not docs:
        return "No relevant information found in the document."
    # combine all matching chunks into one text block
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    print(pdf_retriever.invoke({"question": "What is BFS?"}))