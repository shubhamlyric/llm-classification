""" This module contains the main prediction logic. """

from typing import List, Dict
from collections import Counter
import polars as pl
from src.configs import Parameters


def format_similar_items(similar_items: str) -> str:
    """Format the similar items string returned by the agent."""
    return similar_items


def extract_target_values(similar_items: str, target_name: str) -> List[str]:
    """Extract target values from the similar items string."""
    target_values = []
    for item in similar_items.split("\n"):
        if f"{target_name}=" in item:
            target_values.append(item.split("{target_name}=")[1].strip())
    return target_values


def process_and_predict(
    new_data: pl.DataFrame, agent, prompt_template, parameters: Parameters
) -> List[Dict]:
    """Process the new data and make predictions using the agent."""
    predictions = []

    for row in new_data.iter_rows(named=True):
        input_text = row.get("text_description")

        # Agent retrieves similar items
        similar_items = agent.run(f"Find items similar to: {input_text}")

        # Extract target values from similar items
        target_values = extract_target_values(similar_items, parameters.target_column)

        # Perform voting
        vote_result = Counter(target_values).most_common(1)[0]
        most_common_target, vote_count = vote_result

        # Format similar items information
        similar_info = format_similar_items(similar_items)

        # Construct the prompt
        prompt = prompt_template.format(
            input_features=input_text,
            similar_items=similar_info,
            most_common_target=most_common_target,
            vote_count=vote_count,
            total_votes=len(target_values),
        )

        # Get final prediction from the agent
        prediction = agent.run(prompt).strip()

        predictions.append(
            {
                "id": row.get("id", "Unknown"),
                "most_common_target": most_common_target,
                "vote_count": vote_count,
                "total_votes": len(target_values),
                f"{parameters.target_column}_prediction": prediction,
            }
        )

    return predictions
