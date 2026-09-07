# web_search_tool.py

import os                                  # lets us read values from the .env file
from dotenv import load_dotenv             # loads .env into the environment
from tavily import TavilyClient            # Tavily's official Python client
from langchain_core.tools import tool      # turns our function into an agent tool

load_dotenv()   # this reads your .env file and makes TAVILY_API_KEY available

# Step 1: create a Tavily client using your API key
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


# Step 2: this is the actual tool our AI agent will call
@tool
def web_search(query: str) -> str:
    """Search the web for current information not found in the document."""
    try:
        # ask Tavily to search the web and give us a short summary + top results
        response = tavily_client.search(
            query=query,
            max_results=3,          # only get the top 3 results, keep it simple
            include_answer=True     # ask Tavily to also generate a short direct answer
        )

        # Tavily's response is a dictionary. We pull out the parts we want.
        answer = response.get("answer", "")
        results = response.get("results", [])

        # build a simple text summary to hand back to the agent
        output = f"Quick answer: {answer}\n\nSources:\n"
        for r in results:
            output += f"- {r['title']}: {r['url']}\n"

        return output

    except Exception as e:
        return f"Error: web search failed ({e})"


# Step 3: quick test to make sure it works
if __name__ == "__main__":
    result = web_search.invoke({"query": "who won the 2024 US presidential election"})
    print(result)