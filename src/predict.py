""" This module contains the main prediction logic. """

from collections import Counter
import polars as pl
import numpy as np
from src.configs import Parameters
from src.storage.faiss import FaissStorage


def format_similar_items(similar_items: str) -> str:
    """Format the similar items string returned by the agent."""
    return similar_items


def extract_target_values(similar_items, target_column):
    """Extract target values from similar items"""
    target_values = []

    for item in similar_items:
        text = item["text"]
        # Split the text into key-value pairs
        pairs = text.split(", ")
        for pair in pairs:
            key, value = pair.split("=")
            if key == target_column:
                target_values.append(value)
                break

    return target_values


def process_and_predict(
    new_data: pl.DataFrame,
    faiss_storage: FaissStorage,
    prompt_template,
    parameters: Parameters,
    agent,
    original_data: pl.DataFrame,
) -> pl.DataFrame:
    """Process the new data and make predictions using the agent."""
    predictions = []

    for row in new_data.iter_rows(named=True):
        input_text = row.get("text_description")

        # Perform FAISS similarity search
        k = parameters.num_similar_items
        similar_items = faiss_storage.similarity_search(input_text, k)

        # Extract target values from similar items
        target_values = extract_target_values(similar_items, parameters.target_column)

        # Perform voting
        vote_result = Counter(target_values).most_common(1)[0]
        most_common_target, vote_count = vote_result

        # Format similar items for the prompt
        formatted_similar_items = "\n".join([item["text"] for item in similar_items])

        # Construct the prompt
        prompt = prompt_template.format(
            input_features=input_text,
            similar_items=formatted_similar_items,
            most_common_target=most_common_target,
            vote_count=vote_count,
            total_votes=len(target_values),
        )

        # Get final prediction from the agent
        agent_prediction = agent.run(prompt).strip()

        # Ensure the prediction is either 0 or 1
        final_prediction = (
            1 if agent_prediction.lower() in ["1", "true", "yes", "survived"] else 0
        )

        predictions.append(final_prediction)

    # Add the predictions as a new column to the original dataframe
    prediction_column = pl.Series(f"{parameters.target_column}_prediction", predictions)
    result_df = original_data.with_columns(prediction_column)
    # result_df = result_df.drop(columns=['text_description'])
    print("result_df", result_df)

    return result_df


def process_and_predict_multiple(new_data, faiss_storage, prompt_template, parameters, agents, original_data, model_names):
    """Process and predict using multiple models"""
    predictions = {}
    
    for agent, model_name in zip(agents, model_names):
        model_predictions = []
        for _, row in new_data.iterrows():
            context = faiss_storage.similarity_search(row['text'], k=5)
            context_str = "\n".join([doc.page_content for doc in context])
            
            prompt = prompt_template.format(context=context_str, question=row['text'])
            response = agent.run(prompt)
            model_predictions.append(response)
        
        predictions[f"prediction_{model_name}"] = model_predictions
    
    # Create a DataFrame with the original data and predictions
    result_df = original_data.copy()
    for model_name, preds in predictions.items():
        result_df[model_name] = preds
    
    return result_df
