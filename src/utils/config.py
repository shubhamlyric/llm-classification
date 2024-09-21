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

def load_config(config_path: str):
    # Implement config loading logic here if needed
    pass
