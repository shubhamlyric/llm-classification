""" This module contains a function to create a prompt template for a dataset. """

from langchain.prompts import PromptTemplate


def generate_prompt_template():
    """Create a prompt template for a dataset."""
    return PromptTemplate(
        input_variables=[
            "input_features",
            "similar_items",
            "most_common_target",
            "vote_count",
            "total_votes",
        ],
        template="""
        Given the following example details:
        {input_features}

        Here are similar examples and their target values:
        {similar_items}

        The most common target value among similar items is: {most_common_target}
        This value appeared {vote_count} times out of {total_votes} similar items.

        Based on this voting result and the similarity of features, what is the most likely target value for this example?
        Provide a concise prediction, considering both the voting result and any relevant feature similarities or differences.
        Your answer must be either 0 or 1, with no explanation or additional text.

        Prediction:
        """,
    )
