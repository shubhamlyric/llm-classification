from src.utils.data_models import InputData, Parameters

# Define your input data and parameters here
INPUT_DATA = InputData(
    input_file="large_input.csv"
)

PARAMETERS = Parameters(
    model_name="gpt-3.5-turbo",
    db_type="pinecone",
    embedding_type="openai",
    dataset_name="large_input",
    target_column="target"
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
