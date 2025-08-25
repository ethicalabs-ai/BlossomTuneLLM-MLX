import mlx.core as mx
import numpy as np
from flwr.common.typing import NDArrays
from mlx.utils import tree_flatten, tree_unflatten


def get_parameters(model) -> NDArrays:
    """
    Extracts the model's trainable parameters and converts them to a list of NumPy arrays for Flower.

    Args:
        model (mlx.nn.Module): The MLX model instance.

    Returns:
        NDArrays: A list of NumPy arrays representing the model's trainable weights.
    """
    # Get the nested dictionary of trainable parameters (e.g., LoRA weights)
    trainable_params = model.trainable_parameters()

    # Ensure all lazy computations are performed before converting.
    mx.eval(trainable_params)

    # Flatten the nested dictionary into a simple list of (key, value) tuples
    # and then extract just the mx.array values.
    params_list = [val for _, val in tree_flatten(trainable_params)]

    # Convert each mx.array into a NumPy array for Flower
    numpy_params = [np.array(p) for p in params_list]

    return numpy_params


def set_parameters(model, parameters: NDArrays):
    """
    Updates the MLX model's trainable parameters from a list of NumPy arrays received from Flower.

    Args:
        model (mlx.nn.Module): The MLX model instance to update.
        parameters (NDArrays): A list of NumPy arrays from the Flower server.
    """
    # Get the model's current trainable parameters to use as a structural template
    reference_params = model.trainable_parameters()

    # Flatten the reference parameters to get the correct keys and structure
    param_keys = [key for key, _ in tree_flatten(reference_params)]

    # Create a list of (key, value) tuples, pairing the correct names
    # with the new parameter arrays received from Flower.
    new_params_flat = [
        (key, mx.array(param)) for key, param in zip(param_keys, parameters)
    ]

    # Reconstruct the nested dictionary structure from the flat list
    new_params_tree = tree_unflatten(new_params_flat)

    # Update the model with the new parameters
    model.update(new_params_tree)
