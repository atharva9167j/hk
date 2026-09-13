"""
HK Flax Framework Bindings (`hk.flax`)
Provides model parameter serialization and deserialization for Flax and Linen matching `safetensors.flax`.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

try:
    import flax
    from flax.core import FrozenDict, freeze, unfreeze
    from flax.traverse_util import flatten_dict, unflatten_dict
    HAS_FLAX = True
except ImportError:
    HAS_FLAX = False

from .jax import save_file, load_file, _check_jax


def _check_flax():
    _check_jax()
    if not HAS_FLAX:
        raise ImportError(
            "Flax is not installed. Please install Flax via 'pip install flax' to use hk.flax."
        )


def _flatten_flax_params(params: Any, prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    """Recursively flattens a nested dictionary or FrozenDict into dot-separated keys."""
    if hasattr(params, "unfreeze"):
        params = params.unfreeze()

    items = {}
    if isinstance(params, dict):
        for k, v in params.items():
            new_key = f"{prefix}{sep}{k}" if prefix else str(k)
            if isinstance(v, dict) or hasattr(v, "unfreeze"):
                items.update(_flatten_flax_params(v, prefix=new_key, sep=sep))
            else:
                items[new_key] = v
    else:
        items[prefix] = params
    return items


def _unflatten_flax_params(flat_dict: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
    """Reconstructs a nested dictionary from dot-separated keys."""
    result: Dict[str, Any] = {}
    for key, value in flat_dict.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


def save_model(
    model_or_variables: Any,
    filename: Union[str, Path, os.PathLike],
    metadata: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> None:
    """
    Saves a Flax model's parameter dictionary or FrozenDict to an .hk container.
    Matches `safetensors.flax.save_model` signature.
    """
    _check_flax()

    variables = model_or_variables
    if hasattr(model_or_variables, "variables"):
        variables = model_or_variables.variables
    elif hasattr(model_or_variables, "params"):
        variables = model_or_variables.params

    flat_dict = _flatten_flax_params(variables)
    save_file(flat_dict, filename, metadata=metadata, **kwargs)


def load_model(
    model_or_variables: Any,
    filename: Union[str, Path, os.PathLike],
    device: Optional[Any] = None,
) -> Any:
    """
    Loads weights from an .hk container into a Flax nested parameter structure.
    Matches `safetensors.flax.load_model` signature.
    """
    _check_flax()
    flat_arrays = load_file(filename, device=device)
    nested = _unflatten_flax_params(flat_arrays)
    return freeze(nested)
