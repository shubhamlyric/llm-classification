"""
Preprocess data, create text descriptions, vectorize data, and store in Pinecone/datastorage layer. 
"""

from typing import Any, Dict
import polars as pl
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.storage import Storage
from src.storage.base import BaseStorage
from src.embedding_models import get_embedding_model


def preprocess_data(
    df: pl.DataFrame, target_column: str, advance: bool = False
) -> pl.DataFrame:
    """
    Preprocess the data
    """
    feature_columns = [col for col in df.columns if col != target_column]

    # Handle missing values
    for col in feature_columns + [target_column]:
        if df[col].dtype == pl.Utf8:
            df = df.with_columns(pl.col(col).fill_null("Unknown"))
        else:
            if not df[col].is_null().all():
                df = df.with_columns(pl.col(col).fill_null(strategy="mean"))

    if advance:
        # Encode categorical variables
        le = LabelEncoder()
        for col in feature_columns:
            if df[col].dtype == pl.Utf8:
                encoded = le.fit_transform(df[col].to_numpy())
                df = df.with_columns(pl.Series(name=col, values=encoded))

        # Normalize numerical features
        scaler = StandardScaler()
        numerical_columns = [
            col
            for col in feature_columns
            if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
        ]
        scaled_data = scaler.fit_transform(df.select(numerical_columns).to_numpy())

        for i, col in enumerate(numerical_columns):
            df = df.with_columns(pl.Series(name=col, values=scaled_data[:, i]))

    text_expr = pl.lit("")
    for col in feature_columns + [target_column]:
        if df[col].dtype == pl.Utf8:
            text_expr = (
                text_expr + pl.lit(col) + pl.lit("=") + pl.col(col) + pl.lit(", ")
            )
        else:
            text_expr = (
                text_expr
                + pl.lit(col)
                + pl.lit("=")
                + pl.col(col).cast(pl.Utf8)
                + pl.lit(", ")
            )

    df = df.with_columns(text_expr.alias("text_description"))

    return df


def process_data(
    input_data: pl.DataFrame,
    target_column: str,
    configs: Dict[str, Any] = None,
    embedding_model: str = "openai",
    db_type: str = "faiss",
    **kwargs,
) -> BaseStorage:
    """Process the data"""
    if configs is None:
        configs = {}
    # Get the total number of rows by reading only the first column
    total_rows = input_data.shape[0]

    chunk_size = configs.get("batch_size", 60)
    processed_rows = 0

    # Process the data in chunks
    for chunk_start in range(0, total_rows, chunk_size):
        chunk_end = min(chunk_size, total_rows - chunk_start)

        # Read a chunk of data
        chunk_df = input_data.slice(chunk_start, chunk_end)

        # Preprocess the data
        processed_chunk = preprocess_data(chunk_df, target_column)

        # Vectorize the data using the specified embedding model
        db = Storage(
            db_type, get_embedding_model(model_type=embedding_model, **kwargs)
        ).get_storage()

        # Store in db
        db.load_data(processed_chunk["text_description"].to_list())

        processed_rows += len(processed_chunk)
        print(f"Processed and stored {processed_rows} out of {total_rows} rows...")

    print(f"Finished processing {processed_rows} rows.")
    return db
