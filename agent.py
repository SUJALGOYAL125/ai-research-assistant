# agent.py

from dotenv import load_dotenv
from langchain.agents import create_agent

from calculator_tool import calculator
from web_search_tool import web_search
from retriever_tool import pdf_retriever

load_dotenv()

# Step 1: put all three tools in one list — the agent picks from these
tools = [pdf_retriever, calculator, web_search]

# Step 2: build the agent — model, tools, and instructions all in one call
agent = create_agent(
    model="google_genai:gemini-3.6-flash",
    tools=tools,
    system_prompt=(
        "You are a helpful assistant. Use the pdf_retriever tool for "
        "questions about the uploaded document, the calculator tool "
        "for math, and the web_search tool for questions about "
        "current events or anything not in the document. "
        "If you answer a question using your own general knowledge "
        "instead of a tool, briefly note that at the end of your answer."
    ),
)


def extract_text(content):
    # content can be a plain string, or a list of content blocks
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # pull out just the "text" pieces, ignore extras like signatures
        return "\n".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return str(content)



if __name__ == "__main__":
    # questions = [
    #     "What is BFS?",                       # should use pdf_retriever
    #     "What is 25 * 4?",                    # should use calculator
    #     "Who won the 2024 US election?",      # should use web_search
    # ]
    # questions = [
    #     "What's the latest news about AI regulation?",
    # ]
    # questions = [
    #   "What's 50% of 200?",
    # ]
    # questions = [
    #   "What is the time complexity of BFS?",
    # ]
    # questions = [
    #   "What's the capital of France?",
    # ]
    # questions = [
    #   "If a graph has 6 vertices, what's 6 squared?",
    # ]
    # questions = [
    #   "What is DFS and how is it different from BFS in Big O terms?",
    # ]
    # questions = [
    #     "asdfghjkl",
    # ]
    questions = [
      "Divide 10 by 0",
    ]
    for q in questions:
      print(f"\n--- Question: {q} ---")
      try:
          result = agent.invoke({"messages": [{"role": "user", "content": q}]})
          print("Answer:", extract_text(result["messages"][-1].content))
      except Exception as e:
          print(f"Error: request failed ({e})")