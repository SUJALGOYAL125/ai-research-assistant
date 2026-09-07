# AI Research Assistant

A simple AI agent that can answer questions about a PDF document, do math, and search the web — all through one chat interface. Built with LangChain, Gemini, Chroma, and Streamlit.

**Live demo:** https://ai-research-assistant-ba5wtuvpcsk6wlvwgkvn26.streamlit.app
*(Note: this app uses a personal free-tier API key with limited daily requests, so it may stop responding if it gets a lot of traffic. See a demo recording below if the live link isn't working.)*

---

## What it does

You upload a PDF, then ask questions in a chat box. Behind the scenes, an AI agent decides which of three tools to use to answer you:

- **PDF Search** — looks up relevant parts of your uploaded document
- **Calculator** — solves math expressions safely
- **Web Search** — looks up current information not in the document

The agent picks the right tool automatically based on your question — you don't have to tell it which one to use.

---

## How it works

```
User uploads PDF
      ↓
PDF is split into chunks → converted into embeddings → stored in a Chroma vector database
      ↓
User asks a question in the chat
      ↓
The agent (powered by Gemini) reads the question and decides:
  - Is this about the document?      → use the PDF Search tool
  - Is this a math question?         → use the Calculator tool
  - Is this about something current? → use the Web Search tool
      ↓
The agent uses the tool, reads the result, and writes a final answer
```

### Tech stack

| Piece | Tool used |
|---|---|
| Language model | Google Gemini (`gemini-3.6-flash`) |
| Embeddings | Google `gemini-embedding-001` |
| Vector database | Chroma |
| Agent framework | LangChain (`create_agent`) |
| Web search | Tavily API |
| Frontend | Streamlit |
| Deployment | Streamlit Cloud |

---

## Project files

| File | What it does |
|---|---|
| `app.py` | The Streamlit web app — chat interface + PDF upload |
| `agent.py` | Builds the agent and connects all three tools |
| `retriever_tool.py` | The PDF search tool |
| `calculator_tool.py` | The calculator tool (safe — doesn't use `eval()`) |
| `web_search_tool.py` | The web search tool (uses Tavily) |
| `ingest.py` | Standalone script to load and index a PDF from the command line |
| `rag_query.py` | Standalone script to test PDF search + answering, without the full agent |

`ingest.py` and `rag_query.py` aren't used by the live app — they're kept as simple debugging tools for testing the PDF pipeline on its own.

---

## Running it locally

**1. Clone the repo and install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Add your API keys**

Create a file called `.env` in the project folder:
```
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
```

- Get a Google API key from [Google AI Studio](https://aistudio.google.com)
- Get a Tavily API key from [tavily.com](https://tavily.com) (free tier available)

**3. Run the app**
```bash
streamlit run app.py
```

This opens the app in your browser at `localhost:8501`.

---

## What I tested

Before calling this done, I ran the agent through a set of tricky questions to check it was actually routing to the right tool, not just answering things itself:

- ✅ Basic questions for each tool (PDF, math, web search) — all routed correctly
- ✅ Word-based math like "What's 50% of 200?" and "Divide 10 by 0" — correctly converted to expressions and handled, including clean error messages
- ✅ Nonsense input ("asdfghjkl") — handled gracefully, no crash
- ✅ A question mixing document content and math ("6 vertices, what's 6 squared?") — the agent even self-corrected after trying `^` (which isn't valid Python syntax for powers) and retried with `**`

## Known limitations

Being upfront about what I found while testing:

- **The agent can answer from its own knowledge instead of using a tool.** For example, asking "What's the capital of France?" made it answer directly rather than searching the web, since it already "knows" the answer. I added an instruction so it now discloses when it does this (*"Note: Answer provided using general knowledge"*), but it doesn't happen 100% of the time — this is a known trade-off with how AI agents decide when to use tools.
- **PDF answers aren't always 100% grounded in the document.** On a harder question about DFS vs. BFS, the agent used the PDF search tool correctly, but also added extra facts from its own training that weren't actually in my document. The answer was accurate, but not fully traceable back to the source — worth knowing if you need strict "only from this document" answers.
- **Uploading a second, different PDF doesn't replace the first one.** Right now, new documents get added alongside old ones in the same database, rather than starting fresh. Fine for a single-document demo, not yet built for multi-document use.

## Possible future improvements

- Clear the database automatically when a new PDF is uploaded
- Add a stricter mode that forces tool use instead of letting the model answer from its own knowledge
- Show which tool was used for each answer, directly in the chat UI
- Support multiple PDFs at once

---

## About this project

This was built as a hands-on project to learn how RAG (Retrieval-Augmented Generation) and AI agents work in practice — from basic document retrieval, to building tools an AI can call, to wiring them together into one working assistant with a real interface.