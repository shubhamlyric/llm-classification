"""
Preprocess data, create text descriptions, vectorize data, and store in Pinecone/datastorage layer. 
"""

from typing import Any, Dict
import polars as pl
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.storage import Storage
from src.storage.base import BaseStorage
from src.embedding_models import get_embedding_model


def preprocess_data(df: pl.DataFrame, target_column: str = None) -> pl.DataFrame:
    """
    Preprocess the data.

    Args:
        df (pl.DataFrame): Input DataFrame.
        target_column (str, optional): The name of the target column. Defaults to None.

    Returns:
        pl.DataFrame: Preprocessed DataFrame.
    """

    all_columns = df.columns
    feature_columns = (
        [col for col in all_columns if col != target_column]
        if target_column
        else all_columns
    )

    # Handle missing values
    fill_null_exprs = []
    for col in feature_columns:
        if df[col].dtype == pl.Utf8:
            fill_null_exprs.append(
                pl.when(pl.col(col).is_null())
                .then("Unknown")
                .otherwise(pl.col(col))
                .alias(col)
            )
        else:
            mean_value = df[col].mean()
            fill_null_exprs.append(
                pl.when(pl.col(col).is_null())
                .then(mean_value)
                .otherwise(pl.col(col))
                .alias(col)
            )

    df = df.with_columns(fill_null_exprs)

    # Encode categorical variables
    categorical_columns = [col for col in feature_columns if df[col].dtype == pl.Utf8]
    encode_exprs = []
    for col in categorical_columns:
        le = LabelEncoder()
        encoded = le.fit_transform(df[col].to_list())
        encode_exprs.append(pl.Series(name=col, values=encoded))
    if encode_exprs:
        df = df.with_columns(encode_exprs)

    # Normalize numerical features
    numerical_columns = [
        col for col in feature_columns
        if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
    ]
    if numerical_columns:
        scaler = StandardScaler()
        numerical_data = df.select(numerical_columns).to_numpy()
        scaled_data = scaler.fit_transform(numerical_data)
        scale_exprs = [
            pl.Series(name=col, values=scaled_data[:, i])
            for i, col in enumerate(numerical_columns)
        ]
        df = df.with_columns(scale_exprs)

    # Build the text_description column
    text_expr = pl.concat_str(
        [
            pl.lit(f"{col}=") + pl.col(col).cast(pl.Utf8)
            for col in feature_columns + ([target_column] if target_column else [])
        ],
        separator=", ",
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

    total_rows = input_data.shape[0]

    chunk_size = configs.get("batch_size", 100)
    processed_rows = 0

    # Initialize the storage and embedding model
    embedding_model = get_embedding_model(model_type=embedding_model, **kwargs)
    db = Storage(db_type, embedding_model).get_storage()

    # Process the data in chunks
    for chunk_start in range(0, total_rows, chunk_size):
        chunk_end = min(chunk_size, total_rows - chunk_start)

        chunk_df = input_data.slice(chunk_start, chunk_end)
        processed_chunk = preprocess_data(chunk_df, target_column)

        # Store in db
        db.load_data(processed_chunk["text_description"].to_list())

        processed_rows += len(chunk_df)
        print(f"Processed and stored {processed_rows} out of {total_rows} rows...")

    print(f"Finished processing {processed_rows} rows.")

    # Check if the database has been initialized
    if db.db is not None:
        print(f"Total entries in database: {len(db.db.docstore._dict)}")
    else:
        print("Database has not been initialized yet.")

    return db
