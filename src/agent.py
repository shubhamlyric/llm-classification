"""
LLM retrieval module
"""

from langchain_openai import OpenAI
from langchain_huggingface import HuggingFacePipeline
from langchain_anthropic import AnthropicLLM
from langchain.agents import Tool, initialize_agent
from src.configs import Parameters

OPENAI_MODELS = ["gpt", "gpt-4"]


def create_agent(vectorstore, llm):
    """
    Create a search agent
    """
    search_tool = Tool(
        name="similaritySearch",
        func=lambda query: vectorstore.similarity_search(query, k=1000),
        description="Use this tool to find similar entities based on their features.",
    )
    return initialize_agent(
        tools=[search_tool], llm=llm, agent="zero-shot-react-description", verbose=True
    )


def get_llm(parameters: Parameters):
    """
    Get the LLM model
    """
    if parameters.llm_model_name.lower() in OPENAI_MODELS:
        return OpenAI(
            model=parameters.llm_model_name,
            temperature=parameters.temperature,
            max_tokens=parameters.max_tokens,
        )
    elif parameters.llm_model_name.lower().startswith("claude"):
        return AnthropicLLM(
            model=parameters.llm_model_name,
            temperature=parameters.temperature,
            max_tokens=parameters.max_tokens,
        )
    else:
        return HuggingFacePipeline.from_model_id(
            model_id="gpt2",
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": parameters.max_tokens},
        )
