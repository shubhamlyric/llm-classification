"""
Embedding models for different APIs
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings

# Import other necessary libraries for different models


def get_embedding_model(model_name: str = None, model_type="openai", **kwargs):
    """
    Get the embedding model based on the model type
    """
    if model_type == "openai":
        if not model_name:
            model_name = "text-embedding-3-large"
        return OpenAIEmbeddings(model=model_name, **kwargs)
    elif model_type == "claude":
        if not model_name:
            model_name = "claude-3-opus-20240229"
        return ChatAnthropic(model=model_name, **kwargs)
    else:
        model_name = model_name or "all-MiniLM-L6-v2"
        return HuggingFaceEmbeddings(model_name=model_name, **kwargs)
