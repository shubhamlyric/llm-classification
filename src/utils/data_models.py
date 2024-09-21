from pydantic import BaseModel, Field
from typing import Optional

class InputData(BaseModel):
    input_file: str

    class Config:
        arbitrary_types_allowed = True

class Parameters(BaseModel):
    model_name: str = Field(..., description="Name of the LLM model to use")
    db_type: str = Field(..., description="Type of vector database to use")
    embedding_type: str = Field(..., description="Type of embedding to use")
    dataset_name: str = Field(..., description="Name of the dataset being processed")
    target_column: str = Field(..., description="Name of the target column")

    class Config:
        allow_population_by_field_name = True
