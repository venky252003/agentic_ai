from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.adk.models.lite_llm import LiteLlm

from load_dotenv import load_dotenv
load_dotenv()


app_name = "websearch-agent"
user_id = "user_123"
session_id = "session_123"
#model_openai = LiteLlm(model="openai/gpt-4o-mini")

root_agent = LlmAgent(
    name="basic_search_agent",
    description="An agent that can perform web searches using Google Search and answer questions based on the search results.",
    instruction="I can answer your questions by searching the internet. Just ask me anything!",
    model='gemini-3-flash-preview',
    tools=[google_search])
