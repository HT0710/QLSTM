import os
from typing import Dict, List, Tuple, Union

import torch
import yaml


def device_handler(value: str = "auto") -> str:
    """
    Handles the specification of device choice.

    Parameters
    ----------
    value : str, optional
        The device specification. Valid options: ["auto", "cpu", "cuda", "cuda:[device]"], by default "auto"

    Returns
    -------
    str
        The selected device string
    """
    value = value.strip().lower()

    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if value == "gpu" or value.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA device not found.")
        return value if ":" in value else "cuda"

    if value == "cpu":
        return "cpu"

    raise ValueError(
        f'Invalid device option. Valid options: ["auto", "cpu", "cuda", "cuda:[device]"]. Got {value} instead.'
    )


def workers_handler(value: Union[int, float]) -> int:
    """
    Calculate the number of workers based on an input value.

    Parameters
    ----------
    value : Union[int, float]
        The input value to determine the number of workers.
        Int for a specific number.
        Float for a specific portion.
        Set to 0 to use all available cores.

    Returns
    -------
    int
        The computed number of workers for parallel processing
    """
    max_workers = os.cpu_count() or 1  # Fallback to 1 if cpu_count() returns None

    if isinstance(value, float):
        workers = int(max_workers * value)
    else:
        workers = int(value)  # Ensure value is an integer

    if workers == 0:
        return max_workers

    if not 0 <= workers <= max_workers:
        raise ValueError(
            f"Number of workers is out of bounds. Min: 0 | Max: {max_workers}"
        )

    return workers


def tuple_handler(value: Union[int, List[int], Tuple[int]], max_dim: int) -> Tuple[int]:
    """
    Create a tuple with specified dimensions and values.

    Parameters
    ----------
    value : Union[int, List[int], Tuple[int]]
        The value(s) to populate the tuple with.
        - If an integer is provided, a tuple with 'max_dim' elements, each set to this integer, is created.
        - If a tuple or list of integers is provided, it should have 'max_dim' elements.
    max_dim : int
        The desired dimension (length) of the resulting tuple.

    Returns
    -------
    Tuple[int, ...]
        A tuple containing the specified values

    Raises
    ------
    TypeError
        If 'value' is not an integer, tuple, or list.
    ValueError
        If the length of 'value' is not equal to 'max_dim'.
    """
    if isinstance(value, int):
        return tuple([value] * max_dim)

    try:
        value = tuple(value)
        if len(value) != max_dim:
            raise ValueError(
                f"The length of 'value' must be equal to {max_dim}. Got {len(value)} instead."
            )
        return value

    except Exception:
        raise TypeError(
            f"The 'value' parameter must be an int or tuple or list. Got {type(value)} instead."
        )


def yaml_handler(path: str) -> Dict:
    """
    Check and return a dictionary from yaml path.

    Parameters
    ----------
    path : str
        Path to the yaml file.

    Returns
    -------
    Dict
        A dictionary containing the infomation from the given path.

    Raises
    ------
    FileNotFoundError
        If the given path is not found.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r") as f:
        return yaml.safe_load(f)
