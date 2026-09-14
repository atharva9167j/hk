"""
HK PyTorch Integration (`hk.torch`)
Drop-in replacement for `safetensors.torch` providing save_file, load_file, save_model,
load_model, and lazy zero-copy safe_open with slicing.
"""

import json
import mmap
import os
import struct
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .format import (
    MAGIC,
    VERSION_MAJOR,
    VERSION_MINOR,
    ALIGNMENT_BYTES,
    HEADER_SIZE,
    STORAGE_F32,
    STORAGE_F16,
    STORAGE_BF16,
    STORAGE_INT8,
    STORAGE_INT16,
    STORAGE_INT32,
    STORAGE_INT64,
    STORAGE_UINT8,
    STORAGE_UINT16,
    STORAGE_UINT32,
    STORAGE_UINT64,
    STORAGE_F64,
    STORAGE_BOOL,
    STORAGE_DQ4,
    STORAGE_DQ8,
    STORAGE_DQT,
    STORAGE_SPARSE_2_4,
    STORAGE_SPARSE_F16,
    STORAGE_NULL_REF,
    STORAGE_SHARED_REF,
    STORAGE_Q4_0,
    STORAGE_Q8_0,
    STORAGE_Q2_K,
    STORAGE_Q3_K,
    STORAGE_Q4_K,
    STORAGE_Q5_K,
    STORAGE_Q6_K,
    STORAGE_Q8_K,
    TILE_16X16,
    SPARSITY_CSR,
    TORCH_DTYPE_TO_STORAGE,
    STORAGE_TO_TORCH_DTYPE,
)
from .native import (
    NativeHKWriter,
    NativeHKReader,
    native_unpack_2_4,
    is_native_available,
)


def untile_matrix_16x16(tiled: torch.Tensor, original_shape: Tuple[int, int]) -> torch.Tensor:
    M, K = original_shape
    grid_m, grid_k, tm, tk = tiled.shape
    permuted = tiled.permute(0, 2, 1, 3).contiguous()
    dense = permuted.view(grid_m * tm, grid_k * tk)
    return dense[:M, :K]


def unpack_bitmask(packed_data: Union[bytes, memoryview], shape: Tuple[int, ...]) -> torch.Tensor:
    numel = int(np.prod(shape))
    mask_bytes = (numel + 7) // 8
    mask = np.frombuffer(packed_data[:mask_bytes], dtype=np.uint8)
    vals = np.frombuffer(packed_data[mask_bytes:], dtype=np.float16)
    out = np.zeros(numel, dtype=np.float16)
    bits = np.unpackbits(mask)[:numel]
    nonzero_idx = np.where(bits == 1)[0]
    out[nonzero_idx] = vals[:len(nonzero_idx)]
    return torch.from_numpy(out).view(shape)


class HKModel:
    """Lightweight HK Model container handle."""
    def __init__(self, state_dict: Dict[str, torch.Tensor], metadata: Dict[str, str], path: str):
        self.state_dict = state_dict
        self.metadata = metadata
        self.path = path

    def __iter__(self):
        yield self.state_dict
        yield self.metadata

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.state_dict[key]

    def __contains__(self, key: str) -> bool:
        return key in self.state_dict

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def save_file(
    tensors: Optional[Dict[str, torch.Tensor]] = None,
    filename: Optional[Union[str, Path, os.PathLike]] = None,
    metadata: Optional[Dict[str, str]] = None,
    split_index: int = 0,
    split_count: int = 1,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
    **kwargs,
) -> None:
    if tensors is None:
        if state_dict is not None:
            tensors = state_dict
        else:
            raise ValueError("save_file requires tensors or state_dict")

    if filename is None:
        filename = kwargs.get("path", "")
    if not filename:
        raise ValueError("save_file requires a destination filename")

    path_str = str(filename)
    align = kwargs.get("alignment", 4096 if kwargs.get("universal_alignment", False) else 128)
    writer = NativeHKWriter(alignment=align)
    try:
        if kwargs.get("raw_storage", False):
            writer.set_raw_storage(True)
        if split_count > 1:
            writer.set_sharding(split_index, split_count)

        if metadata:
            for k, v in metadata.items():
                if isinstance(v, (dict, list)):
                    writer.add_metadata_json(str(k), json.dumps(v))
                elif isinstance(v, bool):
                    writer.add_metadata_bool(str(k), v)
                elif isinstance(v, int):
                    writer.add_metadata_int(str(k), v)
                elif isinstance(v, float):
                    writer.add_metadata_float(str(k), v)
                else:
                    writer.add_metadata_string(str(k), str(v))
        if split_count > 1 and "split_index" not in (metadata or {}):
            writer.add_metadata_int("split_index", split_index)
            writer.add_metadata_int("split_count", split_count)

        seen_ptrs: Dict[int, str] = {}
        for k, v in tensors.items():
            if not isinstance(v, torch.Tensor):
                raise TypeError(f"Value for key '{k}' must be a torch.Tensor, got {type(v)}")

            ptr = v.data_ptr()
            t_contig = v.detach().cpu().contiguous()
            stype = TORCH_DTYPE_TO_STORAGE.get(v.dtype, STORAGE_F32)

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
                    tensor_data=t_contig,
                    shape=tuple(v.shape),
                    storage_type=stype,
                )
        writer.write_to_file(path_str)
    finally:
        writer.close()


def save_sharded_file(
    tensors: Optional[Dict[str, torch.Tensor]] = None,
    filename_pattern: Optional[Union[str, Path]] = None,
    max_shard_size: Optional[int] = None,
    metadata: Optional[Dict[str, str]] = None,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
    base_path: Optional[Union[str, Path]] = None,
    max_shard_size_bytes: Optional[int] = None,
    save_index_json: bool = True,
    **kwargs,
) -> List[str]:
    """
    Saves tensors across multiple .hk shard files for 70B+ models.
    Sets standard header flag IS_SHARDED, in-header split_index and split_count,
    and automatically emits the standardized index manifest (e.g. model.hk.index.json).
    """
    if tensors is None:
        if state_dict is not None:
            tensors = state_dict
        else:
            raise ValueError("save_sharded_file requires tensors or state_dict")

    target_pattern = filename_pattern or base_path
    if not target_pattern:
        raise ValueError("save_sharded_file requires filename_pattern or base_path")
    pattern_str = str(target_pattern)

    limit = max_shard_size if max_shard_size is not None else max_shard_size_bytes
    if limit is None:
        limit = 4 * 1024 * 1024 * 1024

    shards: List[Dict[str, torch.Tensor]] = []
    current_shard: Dict[str, torch.Tensor] = {}
    current_size = 0

    for name, tensor in tensors.items():
        t_bytes = tensor.numel() * tensor.element_size()
        if current_shard and (current_size + t_bytes > limit):
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[name] = tensor
        current_size += t_bytes

    if current_shard or not shards:
        shards.append(current_shard)

    num_shards = len(shards)
    shard_paths: List[str] = []
    for i in range(num_shards):
        if "{index}" in pattern_str and "{count}" in pattern_str:
            fname = pattern_str.format(index=i + 1, count=num_shards)
        elif "{index}" in pattern_str:
            fname = pattern_str.format(index=i + 1)
        elif pattern_str.endswith(".hk"):
            base = pattern_str[:-3]
            fname = f"{base}-{i + 1:05d}-of-{num_shards:05d}.hk"
        else:
            fname = f"{pattern_str}-{i + 1:05d}-of-{num_shards:05d}.hk"
        shard_paths.append(fname)

    shard_basenames = [os.path.basename(p) for p in shard_paths]

    # Build standardized index manifest weight map
    weight_map: Dict[str, str] = {}
    total_size_bytes = 0
    for i, shard_tensors in enumerate(shards):
        s_base = shard_basenames[i]
        for t_name, t_val in shard_tensors.items():
            weight_map[t_name] = s_base
            total_size_bytes += t_val.numel() * t_val.element_size()

    # Determine standard index manifest path: e.g. model.hk.index.json
    first_path = Path(shard_paths[0])
    parent_dir = first_path.parent
    if pattern_str.endswith(".hk"):
        index_filename = f"{os.path.basename(pattern_str)}.index.json"
    elif "model" in first_path.name:
        index_filename = "model.hk.index.json"
    else:
        index_filename = f"{first_path.stem.split('-')[0]}.hk.index.json"
    index_path = parent_dir / index_filename

    for i, (shard_tensors, path) in enumerate(zip(shards, shard_paths)):
        meta = dict(metadata or {})
        meta["split_index"] = str(i)
        meta["split_count"] = str(num_shards)
        meta["shard_files"] = json.dumps(shard_basenames)
        meta["index_file"] = os.path.basename(index_path)
        meta["total_size_bytes"] = str(total_size_bytes)
        meta["total_tensors"] = str(len(tensors))
        save_file(
            shard_tensors,
            path,
            metadata=meta,
            split_index=i,
            split_count=num_shards,
            **kwargs,
        )

    if save_index_json:
        index_manifest = {
            "metadata": {
                "total_size": total_size_bytes,
                "total_tensors": len(tensors),
                "split_count": num_shards,
                "format": "hk",
                "version": f"{VERSION_MAJOR}.{VERSION_MINOR}.0",
                **(metadata or {}),
            },
            "weight_map": weight_map,
        }
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_manifest, f, indent=2)

    return shard_paths


def _load_single_file(
    filename: Union[str, Path, os.PathLike],
    device: Union[str, torch.device] = "cpu",
    with_residual: bool = True,
) -> Dict[str, torch.Tensor]:
    hk_file = HKFile(filename, framework="pt", device=device, with_residual=with_residual)
    state_dict: Dict[str, torch.Tensor] = {}
    try:
        for name in hk_file.keys():
            state_dict[name] = hk_file.get_tensor(name)
    finally:
        hk_file.close()
    return state_dict


def load_sharded_file(
    filenames: Union[str, List[Union[str, Path]], Path],
    device: Union[str, torch.device] = "cpu",
    with_residual: bool = True,
) -> Dict[str, torch.Tensor]:
    """Loads all shards and merges into a single state dict."""
    if isinstance(filenames, (str, Path)):
        files = [filenames]
    else:
        files = list(filenames)

    merged_state_dict: Dict[str, torch.Tensor] = {}
    for f in files:
        sub_dict = _load_single_file(f, device=device, with_residual=with_residual)
        merged_state_dict.update(sub_dict)
    return merged_state_dict


def load_file(
    filename: Union[str, Path, os.PathLike],
    device: Union[str, torch.device] = "cpu",
    with_residual: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Loads weights from .hk file; automatically links companion shards if file is sharded
    or if an index manifest (model.hk.index.json) is provided or present in directory.
    """
    import re
    path = Path(filename)

    # 1. Directory inspection
    if path.is_dir():
        for candidate in [path / "model.hk.index.json", *path.glob("*.index.json")]:
            if candidate.is_file():
                return load_file(candidate, device=device, with_residual=with_residual)
        hk_files = sorted(path.glob("*.hk"))
        if hk_files:
            return load_file(hk_files[0], device=device, with_residual=with_residual)
        raise FileNotFoundError(f"No .hk or .index.json files found in directory {path}")

    # 2. Index manifest file (e.g. model.hk.index.json)
    if path.is_file() and path.name.endswith(".index.json"):
        with open(path, "r", encoding="utf-8") as f:
            idx_data = json.load(f)
        weight_map = idx_data.get("weight_map", {})
        shard_names = sorted(set(weight_map.values()))
        shard_paths = [path.parent / fn for fn in shard_names]
        return load_sharded_file(shard_paths, device=device, with_residual=with_residual)

    # 3. Direct shard file inspection
    try:
        with HKFile(path) as f:
            is_sharded = getattr(f, "is_sharded", False)
            split_count = getattr(f, "split_count", 1)
            meta = f.metadata()
            shard_files_meta = meta.get("shard_files")
            index_file_meta = meta.get("index_file")
    except Exception:
        is_sharded = False
        split_count = 1
        shard_files_meta = None
        index_file_meta = None

    if is_sharded and split_count > 1:
        parent_dir = path.parent

        # Check if companion index file exists
        if index_file_meta and (parent_dir / index_file_meta).is_file():
            return load_file(parent_dir / index_file_meta, device=device, with_residual=with_residual)
        if (parent_dir / "model.hk.index.json").is_file():
            return load_file(parent_dir / "model.hk.index.json", device=device, with_residual=with_residual)

        shard_paths = []
        if shard_files_meta:
            try:
                filenames_list = json.loads(shard_files_meta)
                shard_paths = [parent_dir / fn for fn in filenames_list]
            except Exception:
                pass

        if not shard_paths:
            m = re.match(r"^(.*?)-(\d+)-of-(\d+)\.hk$", path.name)
            if m:
                prefix, _, total_str = m.groups()
                total = int(total_str)
                shard_paths = [parent_dir / f"{prefix}-{i + 1:05d}-of-{total:05d}.hk" for i in range(total)]

        if shard_paths and all(p.is_file() for p in shard_paths):
            return load_sharded_file(shard_paths, device=device, with_residual=with_residual)

    return _load_single_file(path, device=device, with_residual=with_residual)


def metadata_set(
    filename: Union[str, Path, os.PathLike],
    key: str,
    val: Union[str, int, float, bool, dict, list],
) -> None:
    """Updates or sets a key-value pair directly in the HK container's metadata in-place without rewriting tensors."""
    import json
    path_str = str(filename)
    val_str = json.dumps(val) if isinstance(val, (dict, list)) else str(val)

    if is_native_available():
        from .native import native_metadata_patch_in_place
        if native_metadata_patch_in_place(path_str, key, val_str):
            return

    _patch_metadata_in_place_py(path_str, key, val)


def _patch_metadata_in_place_py(
    filename: str,
    key: str,
    val: Union[str, int, float, bool, dict, list],
) -> None:
    """Pure-Python in-place metadata patcher."""
    import json
    with open(filename, "r+b") as f:
        header_bytes = f.read(128)
        if len(header_bytes) < 128:
            raise ValueError("File too small to be a valid HK container")

        (
            magic, v_maj, v_min, flags, alignment, split_index,
            n_tensors, n_meta, meta_off, meta_sz,
            toc_off, toc_sz, data_off, app_off, chk, split_count, res1
        ) = struct.unpack("<4s H H I H H Q Q Q Q Q Q Q Q Q H 38s", header_bytes)

        if magic != MAGIC:
            raise ValueError("Invalid magic bytes in HK file")

        f.seek(meta_off)
        m_data = f.read(meta_sz)
        m_pos = 0
        existing_items = []

        for _ in range(n_meta):
            if m_pos + 2 > len(m_data):
                break
            klen = struct.unpack_from("<H", m_data, m_pos)[0]
            m_pos += 2
            k = m_data[m_pos:m_pos + klen].decode("utf-8", errors="ignore")
            m_pos += klen
            tag = m_data[m_pos]
            m_pos += 1
            vlen = struct.unpack_from("<I", m_data, m_pos)[0]
            m_pos += 4
            raw_val = m_data[m_pos:m_pos + vlen]
            m_pos += vlen
            existing_items.append((k, tag, raw_val))

        if isinstance(val, bool):
            new_tag = 0x04
            new_val_bytes = b"\x01" if val else b"\x00"
        elif isinstance(val, int):
            new_tag = 0x02
            new_val_bytes = struct.pack("<q", val)
        elif isinstance(val, float):
            new_tag = 0x03
            new_val_bytes = struct.pack("<d", val)
        elif isinstance(val, (dict, list)):
            new_tag = 0x05
            new_val_bytes = json.dumps(val).encode("utf-8")
        else:
            new_tag = 0x01
            new_val_bytes = str(val).encode("utf-8")

        updated_items = []
        found = False
        for k, tag, raw in existing_items:
            if k == key:
                updated_items.append((k, new_tag, new_val_bytes))
                found = True
            else:
                updated_items.append((k, tag, raw))
        if not found:
            updated_items.append((key, new_tag, new_val_bytes))

        new_meta_buf = bytearray()
        for k, tag, raw in updated_items:
            k_bytes = k.encode("utf-8")
            new_meta_buf.extend(struct.pack("<H", len(k_bytes)))
            new_meta_buf.extend(k_bytes)
            new_meta_buf.append(tag)
            new_meta_buf.extend(struct.pack("<I", len(raw)))
            new_meta_buf.extend(raw)

        f.seek(toc_off)
        toc_data = f.read(toc_sz)

        new_meta_sz = len(new_meta_buf)
        needed = 128 + new_meta_sz + toc_sz
        available = data_off - 128

        if needed <= available:
            new_meta_off = 128
            new_toc_off = 128 + new_meta_sz
            f.seek(0)
            new_header = struct.pack(
                "<4s H H I H H Q Q Q Q Q Q Q Q Q H 38s",
                magic, v_maj, v_min, flags, alignment, split_index,
                n_tensors, len(updated_items), new_meta_off, new_meta_sz,
                new_toc_off, toc_sz, data_off, app_off, chk, split_count, res1
            )
            f.write(new_header)
            f.write(new_meta_buf)
            f.write(toc_data)
            pad = data_off - (new_toc_off + toc_sz)
            if pad > 0:
                f.write(b"\x00" * pad)
        else:
            f.seek(0, os.SEEK_END)
            new_meta_off = f.tell()
            f.write(new_meta_buf)
            f.seek(0)
            new_header = struct.pack(
                "<4s H H I H H Q Q Q Q Q Q Q Q Q H 38s",
                magic, v_maj, v_min, flags, alignment, split_index,
                n_tensors, len(updated_items), new_meta_off, new_meta_sz,
                toc_off, toc_sz, data_off, app_off, chk, split_count, res1
            )
            f.write(new_header)

        f.flush()


def load_hk(
    path: str,
    device: str = "cpu",
    with_residual: bool = True,
    use_mmap: bool = True,
    writable: bool = False,
    **kwargs,
) -> HKModel:
    state_dict = load_file(path, device=device, with_residual=with_residual)
    meta: Dict[str, str] = {}
    try:
        reader = NativeHKReader(path)
        for k in ["model_name", "architecture", "format_version"]:
            val = reader.get_metadata_string(k)
            if val is not None:
                meta[k] = val
        reader.close()
    except Exception:
        pass
    return HKModel(state_dict, meta, path)


def save_hk(
    path: str,
    state_dict: Dict[str, torch.Tensor],
    metadata: Optional[Dict[str, str]] = None,
    **kwargs,
) -> None:
    save_file(state_dict, path, metadata=metadata, **kwargs)


def save_model(
    model: nn.Module,
    filename: Union[str, Path, os.PathLike],
    metadata: Optional[Dict[str, str]] = None,
    **kwargs,
) -> None:
    """
    Saves a PyTorch nn.Module's state dict directly into an .hk container.
    Automatically identifies and deduplicates tied weights (`data_ptr`).
    Matches `safetensors.torch.save_model` signature.

    Args:
        model: PyTorch nn.Module.
        filename: Destination path for the .hk file.
        metadata: Optional string-to-string dictionary.
        **kwargs: Optional HK arguments (e.g. quantize_mode, auto_pack_sparse).
    """
    state_dict = model.state_dict()
    save_file(state_dict, filename, metadata=metadata, **kwargs)


def load_model(
    model: nn.Module,
    filename: Union[str, Path, os.PathLike],
    strict: bool = True,
    device: Union[str, torch.device] = "cpu",
) -> Any:
    """
    Loads weights from an .hk container directly into a PyTorch nn.Module.
    Matches `safetensors.torch.load_model` signature.

    Args:
        model: Target PyTorch nn.Module.
        filename: Path to the .hk binary file.
        strict: Whether to strictly enforce that keys in state_dict match model keys.
        device: Target device for loaded tensors.

    Returns:
        NamedTuple (missing_keys, unexpected_keys) returned by model.load_state_dict.
    """
    state_dict = load_file(filename, device=device)
    return model.load_state_dict(state_dict, strict=strict)


class TensorSlice:
    """
    Provides lazy multidimensional slicing over a tensor without loading the entire matrix.
    Compatible with `safetensors` slice interface (`f.get_slice(name)[0:10]`).
    """

    def __init__(self, hk_file: Any, tensor_name: str, tensor_info: Dict[str, Any]):
        self.hk_file = hk_file
        self.tensor_name = tensor_name
        self.info = tensor_info
        self.shape = list(tensor_info["shape"])

    def get_shape(self) -> List[int]:
        """Returns the shape of the tensor."""
        return list(self.shape)

    def __getitem__(self, item: Any) -> Any:
        """Retrieves and slices the tensor in the requested framework representation."""
        t = self.hk_file.get_tensor(self.tensor_name)
        return t[item]


def _open_mmap_shared(filename: Union[str, Path, os.PathLike], access=mmap.ACCESS_COPY):
    """
    Opens and memory-maps a file with full delete sharing (FILE_SHARE_DELETE on Windows).
    Enables true zero-copy access while permitting safe file unlinking/deletion across all platforms.
    """
    path_str = str(filename)
    if os.name == "nt":
        import ctypes
        import msvcrt
        kernel32 = ctypes.windll.kernel32
        GENERIC_READ = 0x80000000
        FILE_SHARE_READ = 1
        FILE_SHARE_WRITE = 2
        FILE_SHARE_DELETE = 4
        OPEN_EXISTING = 3
        FILE_ATTRIBUTE_NORMAL = 0x80
        handle = kernel32.CreateFileW(
            path_str,
            GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            None,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            None,
        )
        if handle == -1 or handle == 0xFFFFFFFFFFFFFFFF:
            raise OSError(ctypes.GetLastError(), f"Failed to open {path_str}")
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY)
        f = os.fdopen(fd, "rb")
        try:
            mm = mmap.mmap(f.fileno(), 0, access=access)
        except Exception:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        f.close()
        return None, mm
    else:
        f = open(path_str, "rb")
        try:
            mm = mmap.mmap(f.fileno(), 0, access=access)
        except Exception:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return f, mm


class HKFile:
    """
    Context manager for lazy inspection, on-demand tensor loading, and slicing of .hk files.
    Drop-in alternative to `safetensors.safe_open` supporting PyTorch, NumPy, and JAX/Flax.
    """

    def __init__(
        self,
        filename: Union[str, Path, os.PathLike],
        framework: str = "pt",
        device: Union[str, torch.device] = "cpu",
        with_residual: bool = True,
    ):
        fw = framework.lower()
        if fw not in ("pt", "torch", "pytorch", "np", "numpy", "jax", "flax"):
            raise ValueError(
                f"Framework '{framework}' is not supported. Supported frameworks: 'pt', 'np', 'jax', 'flax'."
            )

        self.filename = str(filename)
        self.framework = fw
        self.device = str(device) if device is not None else "cpu"
        self.with_residual = with_residual

        self._f, self._mm = _open_mmap_shared(self.filename)
        self.metadata_dict: Dict[str, str] = {}
        self.tensors_info: Dict[str, Dict[str, Any]] = {}
        self.tensor_names: List[str] = []

        self._read_header_and_toc()

    def _read_header_and_toc(self):
        header_bytes = self._mm[:128]
        (
            magic, v_maj, v_min, flags, alignment, split_index,
            n_tensors, n_meta, meta_off, meta_sz,
            toc_off, toc_sz, data_off, app_off, chk, split_count, _
        ) = struct.unpack("<4s H H I H H Q Q Q Q Q Q Q Q Q H 38s", header_bytes)

        if magic != MAGIC or v_maj != VERSION_MAJOR:
            raise ValueError(f"Invalid HK file magic {magic} or version {v_maj}")

        self.flags = flags
        self.split_index = split_index
        self.split_count = split_count
        self.is_sharded = bool(flags & 0x40)

        # Metadata
        if meta_sz > 0:
            m_data = self._mm[meta_off : meta_off + meta_sz]
            m_pos = 0
            for _ in range(n_meta):
                klen = struct.unpack_from("<H", m_data, m_pos)[0]
                m_pos += 2
                key = m_data[m_pos:m_pos + klen].decode("utf-8")
                m_pos += klen
                tag = m_data[m_pos]
                m_pos += 1
                vlen = struct.unpack_from("<I", m_data, m_pos)[0]
                m_pos += 4
                raw_val = m_data[m_pos:m_pos + vlen]
                m_pos += vlen
                if tag == 0x01:
                    self.metadata_dict[key] = raw_val.decode("utf-8")
                elif tag == 0x02:
                    self.metadata_dict[key] = str(struct.unpack("<q", raw_val)[0])
                elif tag == 0x03:
                    self.metadata_dict[key] = str(struct.unpack("<d", raw_val)[0])
                elif tag == 0x04:
                    self.metadata_dict[key] = str(bool(raw_val[0]))
                else:
                    self.metadata_dict[key] = raw_val.decode("utf-8", errors="ignore")

        # TOC
        toc_bytes = self._mm[toc_off : toc_off + toc_sz]
        toc_pos = 0
        for _ in range(n_tensors):
            nlen = struct.unpack_from("<H", toc_bytes, toc_pos)[0]
            toc_pos += 2
            name = toc_bytes[toc_pos:toc_pos + nlen].decode("utf-8")
            toc_pos += nlen

            stype, tlayout, sparse_type, ndim = struct.unpack_from("4B", toc_bytes, toc_pos)
            toc_pos += 4
            shape = struct.unpack_from(f"<{ndim}Q", toc_bytes, toc_pos)
            toc_pos += 8 * ndim

            (
                d_off, d_sz, r_off, r_sz, s_off, s_sz, b_sz, s_ratio
            ) = struct.unpack_from("<Q Q Q Q Q Q H f", toc_bytes, toc_pos)
            toc_pos += 54

            info = {
                "name": name,
                "shape": tuple(shape),
                "storage_type": stype,
                "tile_layout": tlayout,
                "sparsity_type": sparse_type,
                "data_offset": d_off,
                "data_size": d_sz,
                "residual_offset": r_off,
                "residual_size": r_sz,
                "scale_offset": s_off,
                "scale_size": s_sz,
                "block_size": b_sz,
            }
            self.tensors_info[name] = info
            self.tensor_names.append(name)

    def _to_framework(self, arr: np.ndarray, stype: int = STORAGE_F32) -> Any:
        """Converts raw NumPy array to requested framework tensor/array."""
        if self.framework in ("np", "numpy"):
            return arr

        if self.framework in ("jax", "flax"):
            import jax
            import jax.numpy as jnp
            j_arr = jnp.asarray(arr)
            if self.device != "cpu" and self.device:
                try:
                    return jax.device_put(j_arr, self.device)
                except Exception:
                    pass
            return j_arr

        # Default: PyTorch
        t = torch.from_numpy(arr)
        if stype == STORAGE_BF16:
            t = t.view(torch.bfloat16)
        if self.device != "cpu" and self.device:
            t = t.to(self.device)
        return t

    def keys(self) -> List[str]:
        """Returns the list of tensor names stored in the file."""
        return list(self.tensor_names)

    def metadata(self) -> Dict[str, str]:
        """Returns the metadata dictionary stored in the header."""
        return dict(self.metadata_dict)

    def get_tensor(self, name: str) -> Any:
        """Lazily extracts and decodes a single tensor from storage in the requested framework."""
        if name not in self.tensors_info:
            raise KeyError(f"Tensor '{name}' not found in HK file. Available keys: {self.tensor_names}")

        info = self.tensors_info[name]
        st = info["storage_type"]

        # Null Ref
        if st == STORAGE_NULL_REF:
            if self.framework in ("np", "numpy"):
                return np.zeros(info["shape"], dtype=np.float32)
            if self.framework in ("jax", "flax"):
                import jax.numpy as jnp
                return jnp.zeros(info["shape"], dtype=jnp.float32)
            return torch.zeros(info["shape"], dtype=torch.float32, device=self.device)

        # Shared Ref
        if st == STORAGE_SHARED_REF:
            for prev_name in self.tensor_names:
                prev_info = self.tensors_info[prev_name]
                if (prev_info["data_offset"] == info["data_offset"] or tuple(prev_info["shape"]) == tuple(info["shape"])) and prev_info["storage_type"] != STORAGE_SHARED_REF:
                    return self.get_tensor(prev_name)

        # 2:4 Structured Sparse
        if st == STORAGE_SPARSE_2_4:
            raw_data = memoryview(self._mm)[info["data_offset"] : info["data_offset"] + info["data_size"]]
            t = native_unpack_2_4(raw_data, info["shape"])
            if t is not None:
                return self._to_framework(t.detach().cpu().numpy(), stype=st)

        # Bitmask Sparse
        if st == STORAGE_SPARSE_F16:
            raw_data = memoryview(self._mm)[info["data_offset"] : info["data_offset"] + info["data_size"]]
            t = unpack_bitmask(raw_data, info["shape"])
            return self._to_framework(t.detach().cpu().numpy(), stype=st)

        # Quantized types (NF4, DQ8, DQT) via native Zig SIMD reader
        if st in (STORAGE_DQ4, STORAGE_DQ8, STORAGE_DQT):
            reader = NativeHKReader(self.filename)
            try:
                arr = reader.dequantize(name, with_residual=self.with_residual)
                return self._to_framework(arr, stype=st)
            finally:
                reader.close()

        # GGUF Quantization formats & K-Quants
        if st == STORAGE_Q8_0:
            from .quantization import dequantize_q8_0
            raw_data = bytes(memoryview(self._mm)[info["data_offset"] : info["data_offset"] + info["data_size"]])
            t = dequantize_q8_0(raw_data, info["shape"])
            return self._to_framework(t.detach().cpu().numpy(), stype=st)

        if st == STORAGE_Q4_0:
            from .quantization import dequantize_q4_0
            raw_data = bytes(memoryview(self._mm)[info["data_offset"] : info["data_offset"] + info["data_size"]])
            t = dequantize_q4_0(raw_data, info["shape"])
            return self._to_framework(t.detach().cpu().numpy(), stype=st)

        if st == STORAGE_Q4_K:
            from .quantization import dequantize_q4_k
            raw_data = bytes(memoryview(self._mm)[info["data_offset"] : info["data_offset"] + info["data_size"]])
            t = dequantize_q4_k(raw_data, info["shape"])
            return self._to_framework(t.detach().cpu().numpy(), stype=st)

        if st == STORAGE_Q5_K:
            from .quantization import dequantize_q5_k
            raw_data = bytes(memoryview(self._mm)[info["data_offset"] : info["data_offset"] + info["data_size"]])
            t = dequantize_q5_k(raw_data, info["shape"])
            return self._to_framework(t.detach().cpu().numpy(), stype=st)

        if st == STORAGE_Q6_K:
            from .quantization import dequantize_q6_k
            raw_data = bytes(memoryview(self._mm)[info["data_offset"] : info["data_offset"] + info["data_size"]])
            t = dequantize_q6_k(raw_data, info["shape"])
            return self._to_framework(t.detach().cpu().numpy(), stype=st)

        if st == STORAGE_Q2_K:
            from .quantization import dequantize_q2_k
            raw_data = bytes(memoryview(self._mm)[info["data_offset"] : info["data_offset"] + info["data_size"]])
            t = dequantize_q2_k(raw_data, info["shape"])
            return self._to_framework(t.detach().cpu().numpy(), stype=st)

        if st == STORAGE_Q3_K:
            from .quantization import dequantize_q3_k
            raw_data = bytes(memoryview(self._mm)[info["data_offset"] : info["data_offset"] + info["data_size"]])
            t = dequantize_q3_k(raw_data, info["shape"])
            return self._to_framework(t.detach().cpu().numpy(), stype=st)

        if st == STORAGE_Q8_K:
            from .quantization import dequantize_q8_k
            raw_data = bytes(memoryview(self._mm)[info["data_offset"] : info["data_offset"] + info["data_size"]])
            t = dequantize_q8_k(raw_data, info["shape"])
            return self._to_framework(t.detach().cpu().numpy(), stype=st)

        # Dense unquantized storage types (F32, F16, BF16, INT8, INT32, INT64, UINT8, BOOL)
        if st == STORAGE_F16:
            np_dtype = np.float16
        elif st == STORAGE_BF16:
            np_dtype = np.int16
        elif st == STORAGE_INT8:
            np_dtype = np.int8
        elif st == STORAGE_INT16:
            np_dtype = np.int16
        elif st == STORAGE_INT32:
            np_dtype = np.int32
        elif st == STORAGE_INT64:
            np_dtype = np.int64
        elif st == STORAGE_UINT8:
            np_dtype = np.uint8
        elif st == STORAGE_UINT16:
            np_dtype = np.uint16
        elif st == STORAGE_UINT32:
            np_dtype = np.uint32
        elif st == STORAGE_UINT64:
            np_dtype = np.uint64
        elif st == STORAGE_F64:
            np_dtype = np.float64
        elif st == STORAGE_BOOL:
            np_dtype = np.bool_
        else:
            np_dtype = np.float32

        numel = 1
        for dim in info["shape"]:
            numel *= dim

        if self.framework in ("pt", "torch", "pytorch"):
            torch_dtype = STORAGE_TO_TORCH_DTYPE.get(st, torch.float32)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*The given buffer is not writable.*")
                t = torch.frombuffer(self._mm, dtype=torch_dtype, count=numel, offset=info["data_offset"])

            if info["tile_layout"] == TILE_16X16 and len(info["shape"]) == 2:
                M, K = info["shape"]
                pad_m = (16 - (M % 16)) % 16
                pad_k = (16 - (K % 16)) % 16
                tiled_view = t.view((M + pad_m) // 16, (K + pad_k) // 16, 16, 16)
                t = untile_matrix_16x16(tiled_view, (M, K))
            else:
                t = t.view(info["shape"])

            if self.device != "cpu" and self.device:
                t = t.to(self.device)
            return t

        if self.framework in ("np", "numpy"):
            arr = np.frombuffer(self._mm, dtype=np_dtype, count=numel, offset=info["data_offset"])
            if info["tile_layout"] == TILE_16X16 and len(info["shape"]) == 2:
                M, K = info["shape"]
                pad_m = (16 - (M % 16)) % 16
                pad_k = (16 - (K % 16)) % 16
                tiled_view = torch.from_numpy(arr).view((M + pad_m) // 16, (K + pad_k) // 16, 16, 16)
                arr = untile_matrix_16x16(tiled_view, (M, K)).detach().cpu().numpy()
            else:
                arr = arr.reshape(info["shape"])
            return arr

        if self.framework in ("jax", "flax"):
            import jax
            import jax.numpy as jnp
            arr = np.frombuffer(self._mm, dtype=np_dtype, count=numel, offset=info["data_offset"]).reshape(info["shape"])
            j_arr = jnp.asarray(arr)
            if self.device != "cpu" and self.device:
                try:
                    return jax.device_put(j_arr, self.device)
                except Exception:
                    pass
            return j_arr

        arr = np.frombuffer(self._mm, dtype=np_dtype, count=numel, offset=info["data_offset"]).reshape(info["shape"])
        return self._to_framework(arr, stype=st)

    def get_slice(self, name: str) -> TensorSlice:
        """Returns a lazy slice object over the specified tensor."""
        if name not in self.tensors_info:
            raise KeyError(f"Tensor '{name}' not found in HK file.")
        return TensorSlice(self, name, self.tensors_info[name])

    def close(self):
        """Releases the file handle while preserving memory map reference count for tensors."""
        if self._f is not None:
            try:
                self._f.close()
            except Exception:
                pass
            self._f = None
        self._mm = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


class ShardedHKFile:
    """
    Context manager for zero-copy lazy inspection and sliced on-demand tensor loading
    across multi-file sharded .hk containers (70B+ parameter checkpoints).
    Drop-in alternative to `safetensors.safe_open` for multi-file checkpoints.
    """

    def __init__(
        self,
        index_file_or_dir: Union[str, Path, os.PathLike],
        framework: str = "pt",
        device: Union[str, torch.device] = "cpu",
        with_residual: bool = True,
    ):
        path = Path(index_file_or_dir)
        if path.is_dir():
            candidates = [path / "model.hk.index.json", *path.glob("*.index.json")]
            index_candidate = next((c for c in candidates if c.is_file()), None)
            if index_candidate:
                index_path = index_candidate
            else:
                hk_files = sorted(path.glob("*.hk"))
                if hk_files:
                    index_path = hk_files[0]
                else:
                    raise FileNotFoundError(f"No .index.json or .hk files found in directory {path}")
        else:
            index_path = path

        self.index_file = Path(index_path)
        self.parent_dir = self.index_file.parent
        self.framework = framework.lower()
        self.device = device
        self.with_residual = with_residual

        self.metadata_dict: Dict[str, str] = {}
        self.weight_map: Dict[str, str] = {}
        self.tensor_names: List[str] = []
        self._shard_handles: Dict[str, HKFile] = {}

        self.is_sharded = True
        self.split_index = 0
        self.split_count = 1

        if self.index_file.name.endswith(".index.json"):
            with open(self.index_file, "r", encoding="utf-8") as f:
                idx_data = json.load(f)
            self.metadata_dict = {str(k): str(v) for k, v in idx_data.get("metadata", {}).items()}
            self.weight_map = dict(idx_data.get("weight_map", {}))
            self.tensor_names = list(self.weight_map.keys())
            self.split_count = int(idx_data.get("metadata", {}).get("split_count", len(set(self.weight_map.values())) or 1))
        else:
            primary = self._get_shard_handle(self.index_file.name)
            self.metadata_dict = primary.metadata()
            self.split_index = getattr(primary, "split_index", 0)
            self.split_count = getattr(primary, "split_count", 1)
            shard_files_meta = self.metadata_dict.get("shard_files")
            if shard_files_meta:
                try:
                    shard_list = json.loads(shard_files_meta)
                    for sf in shard_list:
                        h = self._get_shard_handle(sf)
                        for t_name in h.keys():
                            self.weight_map[t_name] = sf
                    self.tensor_names = list(self.weight_map.keys())
                    self.split_count = len(shard_list)
                except Exception:
                    pass
            if not self.tensor_names:
                for t_name in primary.keys():
                    self.weight_map[t_name] = self.index_file.name
                self.tensor_names = list(self.weight_map.keys())

    def _get_shard_handle(self, shard_name: str) -> HKFile:
        if shard_name not in self._shard_handles:
            s_path = self.parent_dir / shard_name
            if not s_path.is_file():
                s_path = Path(shard_name)
            self._shard_handles[shard_name] = HKFile(
                s_path,
                framework=self.framework,
                device=self.device,
                with_residual=self.with_residual,
            )
        return self._shard_handles[shard_name]

    def keys(self) -> List[str]:
        """Returns the complete list of tensor names across all shards."""
        return list(self.tensor_names)

    def metadata(self) -> Dict[str, str]:
        """Returns the metadata dictionary."""
        if not self.metadata_dict and self.weight_map:
            first_shard = next(iter(self.weight_map.values()))
            return self._get_shard_handle(first_shard).metadata()
        return dict(self.metadata_dict)

    def get_tensor(self, name: str) -> Any:
        """Lazily routes and decodes the tensor from its owning shard file."""
        if name not in self.weight_map:
            raise KeyError(f"Tensor '{name}' not found in sharded index. Available keys: {len(self.tensor_names)}")
        shard_name = self.weight_map[name]
        return self._get_shard_handle(shard_name).get_tensor(name)

    def get_slice(self, name: str) -> TensorSlice:
        """Returns a lazy slice object over the specified tensor from its owning shard."""
        if name not in self.weight_map:
            raise KeyError(f"Tensor '{name}' not found in sharded index.")
        shard_name = self.weight_map[name]
        return self._get_shard_handle(shard_name).get_slice(name)

    def close(self):
        """Closes all opened shard file handles."""
        for h in self._shard_handles.values():
            try:
                h.close()
            except Exception:
                pass
        self._shard_handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


def safe_open(
    filename: Union[str, Path, os.PathLike],
    framework: str = "pt",
    device: Union[str, torch.device] = "cpu",
) -> Union[HKFile, ShardedHKFile]:
    """
    Opens an .hk file or sharded index manifest (.index.json) for zero-copy inspection
    and lazy tensor fetching/slicing across PyTorch, NumPy, and JAX/Flax.
    Drop-in alternative to `safetensors.safe_open`.

    Args:
        filename: Path to .hk file, model.hk.index.json manifest, or directory.
        framework: "pt" / "torch", "np" / "numpy", or "jax" / "flax".
        device: Target execution device.

    Usage:
        with safe_open("model.hk.index.json", framework="jax") as f:
            for key in f.keys():
                arr = f.get_tensor(key)
            slice_arr = f.get_slice("embed_tokens.weight")[:10, :]
    """
    path = Path(filename)
    if path.is_dir() or (path.is_file() and path.name.endswith(".index.json")):
        return ShardedHKFile(filename, framework=framework, device=device)

    return HKFile(filename, framework=framework, device=device)
