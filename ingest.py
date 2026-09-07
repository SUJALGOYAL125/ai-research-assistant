from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()  # <-- reads your .env file and loads GOOGLE_API_KEY into the environment

# 1. Load PDF
loader = PyPDFLoader("test.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} pages")

# 2. Split document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Split into {len(chunks)} chunks")

# 3. Create embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# 4. Store embeddings in Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Embeddings stored successfully!")

# 5. Test retrieval — confirm it actually works before moving on
query = "What is BFS?"
results = vectorstore.similarity_search(query, k=1)
print(f"\nQuery: {query}")
print(f"Top result:\n{results[0].page_content}")