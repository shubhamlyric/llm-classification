from langchain_openai import OpenAI
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from src.configs import Parameters
from src.utils.device import get_device
import os
from dotenv import load_dotenv
load_dotenv()

def get_llm(parameters: Parameters, model_name: str):
    device = get_device()
    
    provider, model = model_name.split(':')
    
    if provider == "openai":
        # get openai api key from .env
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key is None:
            raise ValueError("OpenAI API key is not set")
        else:
            print("OpenAI API key is set to: ", openai_api_key)

            openai_org_id = os.getenv("OPENAI_ORG_ID")
            if openai_org_id is None:
                raise ValueError("OpenAI organization ID is not set")
            else:
    
                return OpenAI(
                    model_name=model,
            temperature=parameters.temperature,
            max_tokens=parameters.max_tokens,
        )
    elif provider == "anthropic":
        # get anthropic api key from .env
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_api_key is None:
            raise ValueError("Anthropic API key is not set")
        return ChatAnthropic(
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