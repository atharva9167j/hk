"""
HK JAX Framework Bindings (`hk.jax`)
Provides zero-copy, direct JAX array persistence and ingestion matching `safetensors.jax`.
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

from .numpy import save_file as np_save_file, load_file as np_load_file


def _check_jax():
    if not HAS_JAX:
        raise ImportError(
            "JAX is not installed. Please install JAX via 'pip install jax jaxlib' to use hk.jax."
        )


def save_file(
    tensors: Dict[str, Any],
    filename: Union[str, Path, os.PathLike],
    metadata: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> None:
    """
    Saves a dictionary of JAX arrays directly into an .hk container.
    Matches `safetensors.jax.save_file` signature.
    """
    _check_jax()
    if not isinstance(tensors, dict):
        raise TypeError(f"tensors must be a dict of JAX arrays, got {type(tensors)}")

    np_tensors = {}
    for k, v in tensors.items():
        if isinstance(v, np.ndarray):
            np_tensors[k] = v
        else:
            np_tensors[k] = np.asarray(v)

    np_save_file(np_tensors, filename, metadata=metadata, **kwargs)


def load_file(
    filename: Union[str, Path, os.PathLike],
    device: Optional[Any] = None,
    with_residual: bool = True,
) -> Dict[str, Any]:
    """
    Loads weights from an .hk container directly into a dictionary of JAX arrays.
    Matches `safetensors.jax.load_file` signature.
    """
    _check_jax()
    np_dict = np_load_file(filename, with_residual=with_residual)
    jax_dict = {}
    for k, v in np_dict.items():
        j_arr = jnp.asarray(v)
        if device is not None:
            j_arr = jax.device_put(j_arr, device)
        jax_dict[k] = j_arr
    return jax_dict
