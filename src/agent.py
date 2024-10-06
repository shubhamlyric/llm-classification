"""
LLM retrieval module
"""

from langchain.agents import AgentType, Tool, initialize_agent
from langchain.prompts import PromptTemplate
from langchain_anthropic import AnthropicLLM
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.configs import Parameters
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from src.utils.device import get_device

OPENAI_MODELS = ["gpt", "gpt-4"]


def create_agent(vectorstore, llm):
    """
    Create a search agent
    """
    prompt = PromptTemplate(
        input_variables=[
            "input_features",
            "similar_items",
            "most_common_target",
            "vote_count",
            "total_votes",
        ],
        template="""
        Given the following example details:
        {input_features}

        Here are similar examples and their target values:
        {similar_items}

        The most common target value among similar items is: {most_common_target}
        This value appeared {vote_count} times out of {total_votes} similar items.

        Based on this voting result and the similarity of features, what is the most likely target value for this example?
        Provide a concise prediction, considering both the voting result and any relevant feature similarities or differences.
        Your answer must be either 0 or 1, with no explanation or additional text.

        Prediction:
        """,
    )

    tools = [
        Tool(
            name="Make Prediction",
            func=lambda x: x,  # This tool doesn't actually do anything, it's just a placeholder
            description="Use this tool to make your final prediction based on the given information. The prediction must be either 0 or 1.",  # noqa
        )
    ]

    return initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=1,  # Limit to one iteration to force a direct answer
        prompt=prompt,
    )


def get_llm(parameters: Parameters):
    """
    Get the LLM model
    """
    device = get_device()

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
