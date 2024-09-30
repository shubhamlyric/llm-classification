"""
Global Configs
"""

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Global Configs
    """

    log_level = "INFO"

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

    model_name: str
    db_type: str
    embedding_type: str
    dataset_name: str
    target_column: str
    dataset_name: str
    temperature: float = 0.8
    max_tokens: int = 8192
