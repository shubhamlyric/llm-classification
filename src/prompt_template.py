""" This module contains a function to create a prompt template for a dataset. """

from langchain.prompts import PromptTemplate

TEAMPLATE = """
Given the following {dataset_name} details:
{{input_features}}

Here are similar {dataset_name}s and their target values:
{{similar_items}}

The most common target value among similar items is: {{most_common_target}}
This value appeared {{vote_count}} times out of {{total_votes}} similar items.

Based on this voting result and the similarity of features, what is the most likely target value for this {dataset_name}?
Please provide a concise prediction, considering both the voting result and any relevant feature similarities or differences. The answer should be a single value.
"""


def create_prompt_template(dataset_name: str):
    """Create a prompt template for a dataset."""
    return PromptTemplate(
        input_variables=[
            "input_features",
            "similar_items",
            "most_common_target",
            "vote_count",
            "total_votes",
        ],
        template=TEAMPLATE.replace("{dataset_name}", dataset_name),
    )
