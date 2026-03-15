from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionServices
from google.genai import types
from google.adk.models.lite_llm import LiteLlm

from load_dotenv import load_dotenv
load_dotenv()

app_name = "websearch-agent"
user_id = "user_123"
session_id = "session_123"
model_openai = LiteLlm(model="openai/gpt-4o-mini")

search_agent = LlmAgent(
    name="basic_search_agent",
    description="An agent that can perform web searches using Google Search and answer questions based on the search results.",
    instruction="I can answer your questions by searching the internet. Just ask me anything!",
    model=model_openai,
    tools=[google_search])

async def setup_session_and_runner():
    session_services = InMemorySessionServices()
    session = await session_services.create_session(app_name=app_name, user_id=user_id, session_id=session_id)
    runner = Runner(agent=search_agent, app_name=app_name, session_service=session_services)
    return session, runner

async def call_agent_async(query: str):
    content = types.Content(role='user', parts=[types.Part(text=query)])
    session, runner = await setup_session_and_runner()
    events = await runner.run_async(user_id=user_id, session_id=session_id, new_message=content)
    print("Agent Response:", events)

    async for event in events:
        if event.is_final_response():
            final_response = event.content.parts[0].text
            print("Final Response:", final_response)
            return final_response
        
if __name__ == "__main__":
    import asyncio
    query = "What is latest news on Claude?"
    asyncio.run(call_agent_async(query))
