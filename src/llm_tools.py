"""
LLM retrieval module
"""

from langchain_anthropic import AnthropicLLM
from langchain_huggingface import HuggingFacePipeline, HuggingFaceHub
from langchain_openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.configs import Parameters
from src.utils.device import get_device

OPENAI_MODELS = ["gpt", "gpt-4"]


def get_llm(parameters, model_name):
    """Get the LLM based on the model name"""
    provider, model = model_name.split(':')
    
    if provider == "huggingface":
        return HuggingFaceHub(repo_id=model, model_kwargs={"temperature": float(parameters.temperature)})
    elif provider == "openai":
        return OpenAI(model_name=model, temperature=float(parameters.temperature))
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
