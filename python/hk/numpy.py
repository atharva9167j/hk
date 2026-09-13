"""
HK NumPy Framework Bindings (`hk.numpy`)
Provides zero-copy, direct NumPy array persistence and ingestion matching `safetensors.numpy`.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .format import (
    STORAGE_F32,
    STORAGE_F16,
    STORAGE_INT8,
    STORAGE_INT32,
    STORAGE_INT64,
    STORAGE_UINT8,
    STORAGE_BOOL,
    STORAGE_SHARED_REF,
    STORAGE_NULL_REF,
    STORAGE_DQ4,
    STORAGE_DQ8,
    STORAGE_DQT,
    STORAGE_SPARSE_2_4,
    STORAGE_SPARSE_F16,
    TILE_16X16,
    untile_matrix_16x16,
)
from .native import NativeHKWriter, NativeHKReader, is_native_available

NUMPY_DTYPE_TO_STORAGE = {
    np.dtype("float32"): STORAGE_F32,
    np.dtype("float16"): STORAGE_F16,
    np.dtype("int8"): STORAGE_INT8,
    np.dtype("int32"): STORAGE_INT32,
    np.dtype("int64"): STORAGE_INT64,
    np.dtype("uint8"): STORAGE_UINT8,
    np.dtype("bool"): STORAGE_BOOL,
}

STORAGE_TO_NUMPY_DTYPE = {
    STORAGE_F32: np.float32,
    STORAGE_F16: np.float16,
    STORAGE_INT8: np.int8,
    STORAGE_INT32: np.int32,
    STORAGE_INT64: np.int64,
    STORAGE_UINT8: np.uint8,
    STORAGE_BOOL: np.bool_,
}


def save_file(
    tensors: Dict[str, np.ndarray],
    filename: Union[str, Path, os.PathLike],
    metadata: Optional[Dict[str, str]] = None,
    split_index: int = 0,
    split_count: int = 1,
    **kwargs: Any,
) -> None:
    """
    Saves a dictionary of NumPy ndarrays directly into an .hk container.
    Matches `safetensors.numpy.save_file` signature.
    """
    if not isinstance(tensors, dict):
        raise TypeError(f"tensors must be a dict of NumPy arrays, got {type(tensors)}")

    path_str = str(filename)
    writer = NativeHKWriter(alignment=kwargs.get("alignment", 128))
    try:
        if split_count > 1:
            writer.set_sharding(split_index, split_count)

        if metadata:
            for k, v in metadata.items():
                writer.add_metadata_string(str(k), str(v))
        if split_count > 1 and "split_index" not in (metadata or {}):
            writer.add_metadata_int("split_index", split_index)
            writer.add_metadata_int("split_count", split_count)

        seen_ptrs: Dict[int, str] = {}
        for k, v in tensors.items():
            if not isinstance(v, np.ndarray):
                raise TypeError(f"Value for key '{k}' must be a numpy.ndarray, got {type(v)}")

            v_contig = np.ascontiguousarray(v)
            ptr = v_contig.__array_interface__["data"][0]
            stype = NUMPY_DTYPE_TO_STORAGE.get(v.dtype, STORAGE_F32)

            if ptr in seen_ptrs and v.shape == tensors[seen_ptrs[ptr]].shape and v.dtype == tensors[seen_ptrs[ptr]].dtype:
                writer.add_tensor(
                    name=k,
                    tensor_data=b"",
                    shape=tuple(v.shape),
                    storage_type=STORAGE_SHARED_REF,
                )
            else:
                seen_ptrs[ptr] = k
                writer.add_tensor(
                    name=k,
                    tensor_data=v_contig.tobytes(),
                    shape=tuple(v.shape),
                    storage_type=stype,
                )
        writer.write_to_file(path_str)
    finally:
        writer.close()


def load_file(
    filename: Union[str, Path, os.PathLike],
    with_residual: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Loads weights from an .hk file directly into a dictionary of NumPy ndarrays.
    Automatically handles companion shards or index manifests.
    Matches `safetensors.numpy.load_file` signature.
    """
    from .torch import HKFile, load_file as torch_load_file

    path = Path(filename)

    # Check for index manifest
    if path.is_file() and path.name.endswith(".index.json"):
        tensors_torch = torch_load_file(path, device="cpu", with_residual=with_residual)
        return {k: v.detach().cpu().numpy() for k, v in tensors_torch.items()}

    # Check if sharded
    try:
        with HKFile(path, framework="pt") as f:
            is_sharded = getattr(f, "is_sharded", False)
            split_count = getattr(f, "split_count", 1)
    except Exception:
        is_sharded = False
        split_count = 1

    if is_sharded and split_count > 1:
        tensors_torch = torch_load_file(path, device="cpu", with_residual=with_residual)
        return {k: v.detach().cpu().numpy() for k, v in tensors_torch.items()}

    reader = NativeHKReader(str(path))
    arrays: Dict[str, np.ndarray] = {}
    shared_tensors: List[Tuple[str, Any]] = []

    try:
        for name, meta in reader.tensors.items():
            stype = meta["storage_type"]
            shape = meta["shape"]

            if stype == STORAGE_SHARED_REF:
                shared_tensors.append((name, meta))
                continue

            raw_bytes = reader.get_raw_data(name)
            if stype in STORAGE_TO_NUMPY_DTYPE:
                dtype = STORAGE_TO_NUMPY_DTYPE[stype]
                arr = np.frombuffer(raw_bytes, dtype=dtype).reshape(shape).copy()
            else:
                arr = reader.dequantize(name, with_residual=with_residual)

            arrays[name] = arr

        for shared_name, shared_meta in shared_tensors:
            matched = False
            for primary_name, primary_arr in arrays.items():
                if tuple(primary_arr.shape) == shared_meta["shape"]:
                    arrays[shared_name] = primary_arr
                    matched = True
                    break
            if not matched:
                arrays[shared_name] = np.zeros(shared_meta["shape"], dtype=np.float32)
    finally:
        reader.close()

    return arrays
