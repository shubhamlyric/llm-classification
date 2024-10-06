""" run the pipeline """

import polars as pl
from src.configs import Parameters
from src.data_processing import process_data, preprocess_data
from src.agent.base import BaseAgent
from src.llm_tools import get_llm
from src.predict import process_and_predict
from src.prompt_template import generate_prompt_template


def run(inputs, parameters, configs):
    """Run the pipeline"""
    parameters = Parameters(**parameters)

    vectorstore = process_data(
        input_data=inputs.get("historic"),
        target_column=parameters.target_column,
        configs=configs,
        db_type=parameters.db_type,
        embedding_model=parameters.embedding_type,
    )

    llm = get_llm(parameters)
    agent = BaseAgent(
        parameters=parameters,
        vectorstore=vectorstore,
        llm=llm,
        prompt=generate_prompt_template(),
    ).create_agent()

    # Process the new data
    to_predict_data = inputs.get("to_predict")
    processed_new_data = preprocess_data(
        df=to_predict_data,
        target_column=None,  # We don't have the target column for prediction data
    )

    # Make predictions on the processed new data
    predictions_df = process_and_predict(
        new_data=processed_new_data,
        faiss_storage=vectorstore,
        prompt_template=generate_prompt_template(),
        parameters=parameters,
        agent=agent,
        original_data=to_predict_data,
    )

    return {"predictions": predictions_df}


if __name__ == "__main__":
    params = {
        "llm_model_name": "hugging",
        "db_type": "faiss",
        "embedding_type": "sentence-transformers/paraphrase-MiniLM-L6-v2",
        "dataset_name": "Titanic Survival Passenger",
        "target_column": "Survived",
        "temperature": "0.8",
        "max_tokens": "8192",
    }

    data = {
        "historic": pl.read_csv("data/historic.csv"),
        "to_predict": pl.read_csv("data/to_predict_small.csv"),
    }
    outputs = run(inputs=data, parameters=params, configs={})

    for key, value in outputs.items():
        if isinstance(value, pl.DataFrame):
            value.write_csv(f"data/{key}.csv")
        else:
            print(f"Warning: {key} is not a DataFrame. Skipping CSV write.")
