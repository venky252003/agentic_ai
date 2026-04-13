from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(model="gemma4:latest", model_provider="ollama", temperature=0.9)

def demo_str_parser():
    prompt = ChatPromptTemplate.from_template(
        "Give me a short description of {topic} in less than 20 words."
    )
    praser = StrOutputParser()
    chain = prompt | model | praser
    result = chain.invoke({"topic": "Artificial Intelligence"})
    print(f"StrOutputParser result: {result} (type: {type(result).__name__})")

def demo_json_parser():
    prompt = ChatPromptTemplate.from_template(
        "Give me a short description of {topic} in less than 20 words, and also provide a list of 3 related keywords. Return the result as JSON with keys 'description' and 'keywords'."
    )
    parser = JsonOutputParser()
    chain = prompt | model | parser
    result = chain.invoke({"topic": "Artificial Intelligence"})
    print(f"JsonOutputParser result: {result} (type: {type(result).__name__})")

def demo_pydantic_parser():
    class TopicInfo(BaseModel):
        description: str = Field(..., description="A short description of the topic in less than 20 words.")
        keywords: List[str] = Field(..., description="A list of 3 related keywords.")

    prompt = ChatPromptTemplate.from_template(
        "Give me a short description of {topic} in less than 20 words, and also provide a list of 3 related keywords. Return the result as JSON with keys 'description' and 'keywords'."
    )
    parser = PydanticOutputParser(pydantic_object=TopicInfo)
    chain = prompt | model | parser
    result = chain.invoke({"topic": "Artificial Intelligence"})
    print(f"PydanticOutputParser result: {result} (type: {type(result).__name__})")

def movie_structured_extraction():
    class MovieInfo(BaseModel):
        title: str = Field(description="The title of the movie.")
        year: int = Field(description="The release year of the movie.")
        actors: List[str] = Field(description="A list of main actors in the movie.")
        genre: str = Field(description="The genre of the movie.")
        rating: int = Field(description="The movie's rating on a scale of 1 to 10.", ge=1, le=10)

    structured_model = model.with_structured_output(MovieInfo)
    prompt = ChatPromptTemplate.from_template(
        "Provide structured information about the movie '{movie_name}' including title, release year, main actors, genre, and rating (1-10)."
    )

    chain = prompt | structured_model
    result = chain.invoke({"movie_name": "Inception"})

    print(f"Structured Movie Info: {result} (type: {type(result).__name__})")
    print(f"Title: {result.title}, Year: {result.year}, Actors: {result.actors}, Genre: {result.genre}, Rating: {result.rating}")


if __name__ == "__main__":
    #demo_str_parser()
    #demo_json_parser()
    #demo_pydantic_parser()
    movie_structured_extraction()