"""
Global Configs
"""

from pydantic import BaseModel, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Global Configs
    """

    log_level: str = "INFO"

    class Config:
        """
        Modal Configs
        """

        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"


class Parameters(BaseModel):
    """
    Parameters
    """

    llm_model_name: str
    db_type: str
    embedding_type: str
    dataset_name: str
    target_column: str
    dataset_name: str
    temperature: float = 0.8
    max_tokens: int = 8192
    num_similar_items: int = 5

    @validator('llm_model_name')
    def validate_model_name(cls, v):
        if ':' not in v:
            raise ValueError("model_name must be in the format 'provider:model'")
        return v

    model_config = {
        'protected_namespaces': ()
    }