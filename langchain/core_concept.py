from dotenv import load_dotenv
import os
load_dotenv()

from langchain_core import __version__ as langchain_core_version
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

def main():
    print("Hello, World!")  

if __name__ == "__main__":
    #demo_basic_chain()
    #demo_batch_execution()
    #demo_streaming_execution()
    market_tagline_generator()

