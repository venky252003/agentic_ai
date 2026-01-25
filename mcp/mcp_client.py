import gradio as gr
from smolagents import MCPClient, CodeAgent, OpenAIModel
from smolagents.models import OpenAIServerModel, ChatMessage

import os
from dotenv import load_dotenv

# Load environment variables from .env filehttp://localhost:7860/gradio_api/mcp/sse
load_dotenv()

def check_env():
    # Read API keys
    openai_api_key = os.getenv("OPENAI_API_KEY")
    open_router_api_key = os.getenv("OPENROUTER_API_KEY")
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")

    # Enable LangChain's tracing to monitor and debug chain executions
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")

    if not open_router_api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env file")

    if not langsmith_api_key:
        raise ValueError("LANGSMITH_API_KEY not found in .env file")

    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY not found in .env file")

    print("✓ Environment loaded successfully")
    print(f"✓ OpenAI API Key: {openai_api_key[:10]}..." if openai_api_key else "✗ OpenAI API Key not found")
    print(f"✓ OpenRouter API Key: {open_router_api_key[:10]}..." if open_router_api_key else "✗ OpenRouter API Key not found")
    print(f"✓ LangSmith API Key: {langsmith_api_key[:10]}..." if langsmith_api_key else "✗ LangSmith API Key not found")
    print(f"✓ Tavily API Key: {tavily_api_key[:10]}..." if tavily_api_key else "✗ Tavily API Key not found")

client = MCPClient({"url": "http://localhost:7860/gradio_api/mcp/sse", "transport": "sse"})
tools = client.get_tools()
print(tools)
model = OpenAIModel(model_id="google/gemma-3-27b-it:free", api_base="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
agent = CodeAgent(model=model, tools=tools)

def run_agent(message: str, history):
     # Prompt that guides the model to either respond directly or trigger the tool+code execution
        decision_prompt = f"""You are a smart AI assistant.

Your job is to decide how to respond to the user's message.

If it's a vague or open-ended prompt (e.g., "Can you help me?", "What's up?"),
respond naturally in plain text — greet the user or ask for more input.

If it's a clear task or query that requires tools, code, or calculations (e.g., "Analyze this", "Summarize this", "What is the sentiment of..."), respond using:
<use_tools>
Thoughts: why you need tools
<code>
# Python code here
# Be sure to print the final result in a clear sentence
</code>

When the message asks for sentiment analysis (e.g., "What is the sentiment of ...", "Analyze this text for sentiment", "Do sentiment analysis of ..."), 
ALWAYS use the sentiment_analysis tool available via the MCP interface.

Think internally and respond appropriately.
Do NOT mention category labels like (A) or (B).
Only return what the user should see.

User message:
\"\"\"{message}\"\"\"
"""
        # Ask the model how to respond based on the user's message
        # Use the model's `.generate()` method to get a response based on the decision prompt
        # as "llm_reply = model.generate(messages=[ChatMessage(role="user", content=decision_prompt)])"
        llm_reply = model.generate(messages=[ChatMessage(role="user", content=decision_prompt)])
       
        # If the model decides that tool + code execution is needed
        if "<use_tools>" in llm_reply.content and "<code>" in llm_reply.content and "</code>" in llm_reply.content:
            # Trigger the full agentic loop: reasoning + tool execution
            response = str(agent.run(message))
        else:
            # Otherwise, return the plain model-generated reply
            response = llm_reply.content.strip()
        
        # Update history and return response with updated state
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        return response, history

if __name__ == "__main__":
    try:
        check_env()


        
        demo = gr.Interface(
            fn=run_agent,
            inputs=[
                gr.Textbox(lines=2, placeholder="Enter your message here..."),
                gr.State([])  # To maintain chat history if needed
            ],
            examples=["Analyze the sentiment of the following text: 'This is awesome'"], # Pre-filled example for testing
            outputs=[gr.Textbox(), gr.State([])],
            title="MCP Client with Code Agent",
            description="Interact with the MCP server using a Code Agent."
        )
        demo.launch(server_name="0.0.0.0", server_port=7861)
    finally:
        client.disconnect()
     