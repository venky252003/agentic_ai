from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
    RunnableSequence,
    RunnableBranch
)

load_dotenv()

#model = init_chat_model(model="gemma4:latest", model_provider="ollama", temperature=0.9)
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)

def demo_bot():
    prompt = ChatPromptTemplate.from_template("Summarize the following text in one sentence: {text}")
    chain = prompt | model | StrOutputParser()
    result = chain.invoke({"text": "The quick brown fox jumps over the lazy dog."})
    print(f"Result: {result}")

def demo_parallel_chain():
    summarize_prompt = ChatPromptTemplate.from_template("Summarize in two sentences: {text}")
    keywords_prompt = ChatPromptTemplate.from_template(
        "Extract 5 keywords in the following text: {text}\nReturn as a comma-separated list."
    )
    sentiment_prompt = ChatPromptTemplate.from_template(
        "What is the sentiment of the following text? {text}"
    )
    
    chain1 = summarize_prompt | model | StrOutputParser()
    chain2 = keywords_prompt | model | StrOutputParser()
    chain3 = sentiment_prompt | model | StrOutputParser()
    
    parallel_chain = RunnableParallel(
        summary = chain1,
        keywords = chain2,
        sentiment = chain3
    )

    text = """
    The new AI features are absolutely incredible! Users are loving the
    faster response times and improved accuracy. However, some have noted
    that the pricing could be more competitive. Overall, the product
    launch has been a massive success with record-breaking adoption rates.
    """
    
    result = parallel_chain.invoke({"text": text})
    print(f"Parallel Chain Result: {result}")
    print("Analysis Results:")
    print("Parallel Analysis Results:")
    print(f"  Summary: {result['summary']}")
    print(f"  Keywords: {result['keywords']}")
    print(f"  Sentiment: {result['sentiment']}")

def demo_passthrough_chain():
    prompt = ChatPromptTemplate.from_template(
        "Original question: {question}\n"
        "Context: {context}\n"
        "Answer the question based on the context."
    )
    def fake_retriever(question):
        return " LangChain was created by Harrison Chase in 2022."
    
    chain = (
        RunnableParallel(
            context = RunnableLambda(fake_retriever),
            question = RunnablePassthrough()
        )
        | RunnableLambda(lambda inputs: {"question": inputs["question"], "context": inputs["context"]})
        | prompt | model | StrOutputParser()
    )
    question = "Who created LangChain?"
    result = chain.invoke({"question": question})
    print(f"Question: {question}")
    print(f"Answer: {result}")

def demo_chain_branching():
    code_prompt = ChatPromptTemplate.from_template(
         "You are a coding expert. Help with: {input}"
    )
    general_prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer: {input}"
    )
    classify_prompt = ChatPromptTemplate.from_template(
        "Classify the following input as 'code' or 'general': {input}\n Return only the classification."
    )
    classifier_chain = classify_prompt | model | StrOutputParser()
    def is_code_question(input_dict):
        classification = classifier_chain.invoke(input_dict)
        return classification.strip().lower() == "code"
    
    branch = RunnableBranch(
            (is_code_question, code_prompt | model | StrOutputParser()),
             general_prompt | model | StrOutputParser())
    questions = [
        "How do I reverse a list in Python?",
        "What is the capital of France?"
    ]
    for q in questions:
        result = branch.invoke({"input": q})
        print(f"Question: {q}")
        print(f"Answer: {result}")

if __name__ == "__main__":
    #demo_bot()
    demo_parallel_chain()