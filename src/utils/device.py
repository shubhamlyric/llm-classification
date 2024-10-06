""" Device utility functions. """

import torch


def get_device():
    """Get the device (GPU or CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
