from src.utils.data_models import InputData, Parameters

# Define your input data and parameters here
INPUT_DATA = InputData(
    input_file="train.csv"
)

PARAMETERS = Parameters(
    model_name="meta-llama/Llama-2-7b-hf",
    db_type="pinecone",
    embedding_type="openai",
    dataset_name="titanic-passenger-survival",
    target_column="Survived"
)

NEW_DATA = InputData(
    input_file="new_data.csv"
)

# Additional configurations
CONFIGS = {
    "batch_size": 32,
    "max_tokens": 150,
    "temperature": 0,
    # Add any other configuration parameters here
}

def get_config():
    return CONFIGS
