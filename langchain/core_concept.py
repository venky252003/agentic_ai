from dotenv import load_dotenv
import os
load_dotenv()

from langchain_core import __version__ as langchain_core_version
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

print(f"LangChain Core Version: {langchain_core_version}")
#print(f"LangGraph Version: {langgraph_version}")

def demo_streaming_execution():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9, streaming=True)
    prompt = ChatPromptTemplate.from_template("Your helpful assistant. Write short details on this {topic}. Total world should be less than 100 words.")
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser
    topic = input("Enter a topic: ")
    answer = chain.invoke({"topic": topic})
    print(f"Answer: {answer}")

def demo_batch_execution():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)
    prompt = ChatPromptTemplate.from_template("Your helpful assistant. Write short details on this {topic}. Total world should be less than 100 words.")
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser
    topics = ["AI", "ML", "DL"]
    answers = chain.batch([{"topic": topic} for topic in topics])
    for topic, answer in zip(topics, answers):
        print(f"Topic: {topic}\nAnswer: {answer}\n")

def demo_basic_chain():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)
    prompt = ChatPromptTemplate.from_template("\
                Your helpful assistant. Write short details on this {topic}. Total world should be less than 100 words." )
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser
    topic = input("Enter a topic: ")
    answer = chain.invoke({"topic": topic})
    print(f"Answer: {answer}")

def market_tagline_generator():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)
    prompt = ChatPromptTemplate.from_template("\
                You are a marketing expert. Generate a catchy tagline for this product: {product} for given target audience: {audience}. Total world should be less than 10 words." )
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser
    product = input("Enter a product name: ")
    audience = input("Enter the target audience: ")
    tagline = chain.invoke({"product": product, "audience": audience})
    print(f"Generated Tagline: {tagline}")

def new_chain():
    llm = init_chat_model(model="qwen3.5:9b", model_provider="ollama", temperature=0.9)
    prompt = ChatPromptTemplate.from_template("\
                Your helpful assistant. Write short details on this {topic}. Total world should be less than 100 words.")
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser
    topic = input("Enter a topic: ")
    poem = chain.invoke({"topic": topic})
    print(f"Generated Poem: {poem}")

def demo_message_chain():
    llm = init_chat_model(model="gemma4:latest", model_provider="ollama", temperature=0.9)
    output_parser = StrOutputParser()

    chain = [SystemMessage(content="You are a helpful assistant. Write short details on the given topic. Total world should be less than 100 words."),
             HumanMessage(content="{topic}")]
    
    chain = ChatPromptTemplate.from_messages(chain) | llm | output_parser
    topic = input("Enter a topic: ")
    answer = chain.invoke({"topic": topic})
    print(f"Answer: {answer}")

def main():
    print("Hello, World!")  

if __name__ == "__main__":
    #demo_basic_chain()
    #demo_batch_execution()
    #demo_streaming_execution()
    #market_tagline_generator()
    #new_chain()
    demo_message_chain()

