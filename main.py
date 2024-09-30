""" run the pipeline """

import polars as pl
from src.configs import Parameters
from src.data_processing import process_data, preprocess_data
from src.agent import create_agent, get_llm
from src.predict import process_and_predict
from src.prompt_template import create_prompt_template


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
    agent = create_agent(vectorstore=vectorstore, llm=llm)

    # Process the new data
    processed_new_data = preprocess_data(
        df=inputs.get("features"),
        target_column=parameters.target_column,
    )

    # Make predictions on the processed new data
    predictions = process_and_predict(
        new_data=processed_new_data,
        agent=agent,
        prompt_template=create_prompt_template(parameters.dataset_name),
        parameters=parameters,
    )

    # Convert predictions to a Polars DataFrame
    predictions_df = pl.DataFrame(predictions)

    return {"predictions": predictions_df}


if __name__ == "__main__":
    pass
