from langchain_openai import OpenAI
from langchain_anthropic import AnthropicLLM
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from src.configs import Parameters
from src.utils.device import get_device

def get_llm(parameters: Parameters):
    device = get_device()
    
    provider, model = parameters.model_name.split(':')
    
    if provider == "openai":
        return OpenAI(
            model_name=model,
            temperature=parameters.temperature,
            max_tokens=parameters.max_tokens,
        )
    elif provider == "anthropic":
        return AnthropicLLM(
            model=model,
            temperature=parameters.temperature,
            max_tokens=parameters.max_tokens,
        )
    elif provider == "huggingface":
        # Check if it's a local model or needs to be downloaded

        tokenizer = AutoTokenizer.from_pretrained(model)
        llm_model = AutoModelForCausalLM.from_pretrained(model).to(device)

        # Create a pipeline
        pipe = pipeline(
            "text-generation",
            model=llm_model,
            tokenizer=tokenizer,
            max_new_tokens=parameters.max_tokens,
            temperature=parameters.temperature,
            top_p=0.95,
            repetition_penalty=1.15,
            device=device,
        )

        # Create the HuggingFacePipeline instance
        return HuggingFacePipeline(pipeline=pipe)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")