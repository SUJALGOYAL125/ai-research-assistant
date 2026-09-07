# retriever_tool.py

from dotenv import load_dotenv          # ADD THIS
load_dotenv()
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.tools import tool

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