""" run the pipeline """

import polars as pl
from src.configs import Parameters
from src.data_processing import process_data, preprocess_data
from src.agent.base import BaseAgent
from src.llm_tools import get_llm
from src.predict import process_and_predict_multiple
from src.prompt_template import generate_prompt_template


def run(inputs, parameters, configs):
    """Run the pipeline"""
    parameters = Parameters(**parameters)

    vectorstore = process_data(
        input_data=inputs.get("historic"),
        target_column=parameters.target_column,
        configs=configs,
        db_type=parameters.db_type,
        embedding_model=parameters.embedding_type.split(',')[0],  # Use the first embedding model
    )

    # Split the model names and create a list of LLMs
    model_names = parameters.llm_model_name.split(',')
    llms = [get_llm(parameters, model_name) for model_name in model_names]

    agents = []
    for llm in llms:
        agent = BaseAgent(
            parameters=parameters,
            vectorstore=vectorstore,
            llm=llm,
            prompt=generate_prompt_template(),
        ).create_agent()
        agents.append(agent)

    # Process the new data
    to_predict_data = inputs.get("to_predict")
    processed_new_data = preprocess_data(
        df=to_predict_data,
        target_column=None,  # We don't have the target column for prediction data
    )

    # Make predictions on the processed new data using multiple models
    predictions_df = process_and_predict_multiple(
        new_data=processed_new_data,
        faiss_storage=vectorstore,
        prompt_template=generate_prompt_template(),
        parameters=parameters,
        agents=agents,
        original_data=to_predict_data,
        model_names=model_names,
    )

    return {"predictions": predictions_df}


if __name__ == "__main__":
    # params = {
    #     "llm_model_name": "huggingface:meta-llama/Llama-3.2-1B-Instruct,openai:gpt-4-mini,anthropic:claude-3-opus-20240229",
    #     "db_type": "faiss",
    #     "embedding_type": "sentence-transformers/paraphrase-MiniLM-L6-v2,openai:text-embedding-ada-002, anthropic:claude-2",
    #     "dataset_name": "Titanic Survival Passenger",
    #     "target_column": "Survived",
    #     "temperature": "0.8",
    #     "max_tokens": "8192",
    # }
    # params = {
    #     "llm_model_name": "openai:gpt-4-mini,anthropic:claude-3-opus-20240229",
    #     "db_type": "faiss",
    #     "embedding_type": "openai:text-embedding-ada-002,anthropic:claude-2",
    #     "dataset_name": "Titanic Survival Passenger",
    #     "target_column": "Survived",
    #     "temperature": "0.8",
    #     "max_tokens": "8192",
    # }
    params = {
        "llm_model_name": "huggingface:meta-llama/Llama-3.2-1B-Instruct,huggingface:TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "db_type": "faiss",
        "embedding_type": "sentence-transformers/paraphrase-MiniLM-L6-v2,sentence-transformers/all-MiniLM-L6-v2",
        "dataset_name": "Titanic Survival Passenger",
        "target_column": "Survived",
        "temperature": "0.8",
        "max_tokens": "8192",
    }

    data = {
        "historic": pl.read_csv("data/historic.csv"),
        "to_predict": pl.read_csv("data/to_predict_small copy.csv"),
    }
    outputs = run(inputs=data, parameters=params, configs={})

    for key, value in outputs.items():
        if isinstance(value, pl.DataFrame):
            value.write_csv(f"data/{key}.csv")
        else:
            print(f"Warning: {key} is not a DataFrame. Skipping CSV write.")