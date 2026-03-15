from google.adk.agents.llm_agent import Agent
from load_dotenv import load_dotenv
import datetime
load_dotenv()

def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city. """
    return {"status": "success", "city": city, "current_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

root_agent = Agent(
    model='gemini-3-flash-preview',
    name="root_agent",
    description="""Tells the current time in a specified city."""
    instructions="""You are a helpful assistant that can answer questions about the current time in a specified city. You can also answer questions about the weather in a specified city."""
    tools=[get_current_time]
)
