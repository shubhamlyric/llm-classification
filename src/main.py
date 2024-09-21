from src.utils.config import INPUT_DATA, PARAMETERS, NEW_DATA, load_config
from src.retrieval.llm_retrieval import setup_retrieval_and_prediction
from vector_store import VectorStore
import polars as pl

def run(input_data, new_data, parameters, configs):
    # Initialize the vector store
    vs = VectorStore(
        db_type=parameters.db_type,
        embedding_type=parameters.embedding_type
    )
    
    # Set up the retrieval tool, LLM, prompt template, agent, and prediction function
    llm, search_tool, prompt_template, agent, predict_func = setup_retrieval_and_prediction(vs, parameters)
    
    # Process the new data and make predictions
    predictions = predict_func(new_data, agent, prompt_template, parameters)
    
    # Convert predictions to a Polars DataFrame
    predictions_df = pl.DataFrame(predictions)
    
    return {
        "predictions": predictions_df,
        # ... other output data ...
    }

if __name__ == "__main__":
    configs = load_config('config.json')
    new_data = pl.read_csv(NEW_DATA.input_file)  # Assuming NEW_DATA is defined in config.py
    output = run(INPUT_DATA, new_data, PARAMETERS, configs)
    print(output["predictions"])