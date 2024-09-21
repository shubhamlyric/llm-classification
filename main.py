from src.utils.config import INPUT_DATA, PARAMETERS, NEW_DATA, get_config
from src.retrieval.llm_retrieval import setup_retrieval_and_prediction
from src.data_processing.data_processing import process_data
import polars as pl

def run(input_data, new_data, parameters, configs):
    # Process the input data
    
    processed_data = process_data(input_data.input_file, target_column= parameters.target_column, index_name = parameters.dataset_name, configs = configs, embedding_model = parameters.model_name)
    
    # Set up the retrieval tool, LLM, prompt template, agent, and prediction function
    llm, search_tool, prompt_template, agent, predict_func = setup_retrieval_and_prediction(processed_data, parameters)
    
    # Process the new data
    processed_new_data = process_data(new_data.input_file,  target_column= parameters.target_column, index_name = parameters.dataset_name, configs = configs, embedding_model = parameters.model_name)
    
    # Make predictions on the processed new data
    predictions = predict_func(processed_new_data, agent, prompt_template, parameters)
    
    # Convert predictions to a Polars DataFrame
    predictions_df = pl.DataFrame(predictions)
    
    return {
        "predictions": predictions_df,
        # ... other output data ...
    }

if __name__ == "__main__":
    configs = get_config()
    new_data = pl.read_csv(NEW_DATA.input_file)
    output = run(INPUT_DATA, new_data, PARAMETERS, configs)
    print(output["predictions"])