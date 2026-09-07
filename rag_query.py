from langchain_community.vectorstores import Chroma
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------
# 1. Embedding model
# ------------------------------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# ------------------------------------------------
# 2. Load existing Chroma database
# ------------------------------------------------
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# ------------------------------------------------
# 3. Create retriever
# ------------------------------------------------
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

# ------------------------------------------------
# 4. Gemini LLM
# ------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

# ------------------------------------------------
# 5. Prompt
# ------------------------------------------------
prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below.

If the answer is not present in the context,
say "I don't know based on the provided document."

Context:
{context}

Question:
{question}
""")

# ------------------------------------------------
# 6. Question
# ------------------------------------------------
query = "Explain the difference between BFS and DFS"

# ------------------------------------------------
# 7. Retrieve relevant documents
# ------------------------------------------------
docs = retriever.invoke(query)

# ------------------------------------------------
# 8. Combine retrieved chunks
# ------------------------------------------------
context = "\n\n".join(
    doc.page_content for doc in docs
)

# ------------------------------------------------
# 9. Create prompt
# ------------------------------------------------
messages = prompt.invoke({
    "context": context,
    "question": query
})

# ------------------------------------------------
# 10. Ask Gemini
# ------------------------------------------------
response = llm.invoke(messages)

# ------------------------------------------------
# 11. Print answer
# ------------------------------------------------
print("Question:", query)

print("\nAnswer:")

if isinstance(response.content, list):
    print(response.content[0]["text"])
else:
    print(response.content)

print("\n--- Source chunk(s) used ---")

for doc in docs:
    print(doc.page_content[:200])
    print("...")