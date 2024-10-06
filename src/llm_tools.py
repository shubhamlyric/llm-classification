"""
LLM retrieval module
"""

from langchain_anthropic import AnthropicLLM
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.configs import Parameters
from src.utils.device import get_device

OPENAI_MODELS = ["gpt", "gpt-4"]


def get_llm(parameters: Parameters):
    """
    Get the LLM model
    """
    if parameters.llm_model_name.lower() in ["gpt", "gpt-4"]:
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
        # Load the model and tokenizer
        device = get_device()
        model_id = "meta-llama/Llama-3.2-1B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

        # Create a pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=parameters.max_tokens,
            temperature=parameters.temperature,
            top_p=0.95,
            repetition_penalty=1.15,
            device=device,
        )

        # Create the HuggingFacePipeline instance
        return HuggingFacePipeline(pipeline=pipe)
