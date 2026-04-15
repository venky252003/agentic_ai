from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional
from langsmith import traceable
import os

load_dotenv()

if os.getenv("LANGSMITH_API_KEY") is None:
    raise ValueError("LANGSMITH_API_KEY environment variable is not set. Please set it to use tracing features.")
else:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", "smart_bot")
    print(f"LANGSMITH_PROJECT is set to: {os.getenv('LANGSMITH_PROJECT')}")

class SmartBotModel(BaseModel):
    answer: str = Field(description="The answer to the user's question.")
    confidence: str = Field(description="The confidence level of the answer, e.g., 'high', 'medium', 'low'.")
    reasoning: str = Field(description="The reasoning behind the answer, if applicable.")
    follow_up_questions: List[str] = Field(
        description="A list of follow-up questions to ask the user for clarification or additional information.",
        default_factory=list        
    )
    sources_need: bool = Field(description="Whether the answer requires citing sources.", default=False)    

class SmartQABot:
    def __init__(self, model_name: str = "gemma4:latest", temperature: float = 0.7):
        self.model = init_chat_model(model=model_name, model_provider="ollama", temperature=temperature)
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system",  """You are a knowledgeable Q&A assistant.

                Your guidelines:
                - Answer questions accurately and concisely
                - Be honest about uncertainty - set confidence to 'low' if unsure
                - Provide clear reasoning for your answers
                - Suggest relevant follow-up questions
                - Indicate if external sources would help

                Always respond with accurate, helpful information."""),
            ("human", "{question}")
        ])
        self.chain = self.prompt_template | self.model.with_structured_output(SmartBotModel)

    @traceable(name="ask_question", run_type="chain")
    def ask_question(self, question: str) -> SmartBotModel:
        try:
            result = self.chain.invoke({"question": question})
            return result
        except Exception as e:
            print(f"Error during question answering: {e}")
            return SmartBotModel(
                answer="Sorry, I encountered an error while trying to answer your question.",
                confidence="low",
                reasoning=str(e),
                follow_up_questions=[],
                sources_need=False
            )
        
    @traceable(name="ask_batch", run_type="chain")
    def ask_batch(self, questions: List[str]) -> List[SmartBotModel]:
        results = []
        for question in questions:
            result = self.ask_question(question)
            results.append(result)
        return results
    
def demo_bot():
    bot = SmartQABot()
    question = "What are the health benefits of drinking green tea?"
    answer = bot.ask_question(question)
    print(f"Question: {question}")
    print(f"Answer: {answer.answer}")
    print(f"Confidence: {answer.confidence}")
    print(f"Reasoning: {answer.reasoning}")
    print(f"Follow-up Questions: {answer.follow_up_questions}")
    print(f"Sources Needed: {answer.sources_need}")

def demo_batch():
    bot = SmartQABot()
    questions = [
        "What are the health benefits of drinking green tea?",
        "How does photosynthesis work?",
        "What is the capital of France?"
    ]
    answers = bot.ask_batch(questions)
    for q, a in zip(questions, answers):
        print(f"Question: {q}")
        print(f"Answer: {a.answer}")
        print(f"Confidence: {a.confidence}")
        print(f"Reasoning: {a.reasoning}")
        print(f"Follow-up Questions: {a.follow_up_questions}")
        print(f"Sources Needed: {a.sources_need}")
        print("-" * 40)

if __name__ == "__main__":
    #demo_bot()
    demo_batch()