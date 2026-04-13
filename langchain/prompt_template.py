from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage)
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

def demo_basic_templates():
    simple = ChatPromptTemplate.from_template("Translate '{text}' to {language}")
    messages = simple.format_messages(text="Hello, world!", language="French")
    print("Simple template:")
    print(f"  {messages}")

    multi = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a translator. Be concise."),
            ("human", "Translate '{text}' to {language}"),
        ]
    )

    messages = multi.format_messages(text="Good morning", language="Japanese")
    print("\nMulti-message template:")
    for msg in messages:
        print(f"  {type(msg).__name__}: {msg.content}")

def demo_message_types():
    #model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    model = init_chat_model(model="gemma4:latest", model_provider="ollama", temperature=0.9)

    messages = [
        SystemMessage(content="you are a math tutor. Be brief."),
        HumanMessage(content="What's 5 * 5?"),
        AIMessage(content="25"),
        HumanMessage(content="And if I add 10?"),
    ]

    response = model.invoke(messages)
    print(f"Conversation result: {response.content}")

def demo_messages_placeholder():
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Be concise."),
            ("human", "What is the capital of {country}?"),
            MessagesPlaceholder(variable_name="conversation"),
        ]
    )
    conversation = [
        AIMessage(content="The capital of France is Paris."),
        HumanMessage(content="What about India?"),
    ]
    messages = prompt.format_messages(country="France", conversation=conversation)
    print("with history placeholder:")
    for msg in messages:
        print(f"  {type(msg).__name__}: {msg.content}")

    model = init_chat_model(model="gemma4:latest", model_provider="ollama", temperature=0.9)
    chain = prompt | model   

    response = chain.invoke({"country": "India", "conversation": conversation})
    print(f"Response: {response.content}")


def demo_few_shot():
    exmaples = [
        {"country": "France", "capital": "Paris"},
        {"country": "Japan", "capital": "Tokyo"},
        {"country": "India", "capital": "New Delhi"},
    ]
    example_prompt = ChatPromptTemplate.from_messages(
        [
             ("system", "You are a helpful assistant. Be concise."),
             ("human", "What is the capital of {country}?"),
             ("ai", "{capital}"),
        ]
    )

    prompt = FewShotChatMessagePromptTemplate(
        examples=exmaples,
        example_prompt=example_prompt
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Be concise."),
            ("human", "What is the capital of {country}?"),
        ]
    )
    model = init_chat_model(model="gemma4:latest", model_provider="ollama", temperature=0.9)    
    chain = prompt | final_prompt | model
    response = chain.invoke({"country": "India"})
    print(f"Few-shot response: {response.content}")

def demo_prompt_composition():
    persona = ChatPromptTemplate.from_messages(
        [("system", "You are a {role}. Your tone is {tone}.")]
    )
    task = ChatPromptTemplate.from_messages(
        [("human", "{task}")]
    )

    full_prompt = persona + task
    model = init_chat_model(model="gemma4:latest", model_provider="ollama", temperature=0.9)
    chain = full_prompt | model
    response = chain.invoke({"role": "chef", "tone": "friendly", "task": "How do I make hyderabad biryani?"})
    print(f"Composed prompt response: {response.content}")

if __name__ == "__main__":
    #demo_basic_templates()
    #demo_message_types()
    #demo_messages_placeholder()
    #demo_few_shot() 
    demo_prompt_composition()