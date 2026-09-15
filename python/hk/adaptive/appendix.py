"""
HK Adaptive Neural Framework: Appendix Region Manager
Provides reading, appending, cryptographic SHA-256 lineage verification, rollback,
and delta merging for version-chained neural evolution in HK binary containers.
"""

import os
import struct
import hashlib
import time
from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union

class AppendixEntryType(IntEnum):
    LORA_ADAPTER = 0x01
    DELTA_PATCH = 0x02
    NEW_LAYER = 0x03
    CODE_EVAL = 0x04
    KV_CACHE_SINK = 0x05
    TOPOLOGY_HEAD = 0x06

class AppendixFlags(IntEnum):
    ACTIVE = 0x01
    COMPRESSED = 0x02

@dataclass
class AppendixMetrics:
    loss: float = 0.0
    accuracy: float = 0.0
    pass_rate: float = 0.0
    custom: float = 0.0

@dataclass
class AppendixRecord:
    entry_type: AppendixEntryType
    name: str
    generation: int
    data: bytes
    target: str = ""
    flags: int = AppendixFlags.ACTIVE
    timestamp: int = 0
    parent_hash: bytes = b"\x00" * 32
    metrics: AppendixMetrics = field(default_factory=AppendixMetrics)

HEADER_SIZE = 128
REC_HEADER_FORMAT = "<BBHIQ32sffffHHIQ" # 80 bytes (entry_type, flags, name_len, generation, timestamp, parent_hash[32], loss, acc, pass, custom, target_len, reserved, crc32, data_size)
REC_HEADER_SIZE = 80

def compute_parent_hash(data: bytes, name: str = "", target: str = "") -> bytes:
    """Computes SHA-256 digest of record payload, including optional name and target."""
    h = hashlib.sha256()
    if name:
        h.update(name.encode("utf-8"))
    if target:
        h.update(target.encode("utf-8"))
    h.update(data)
    return h.digest()

def read_appendix(file_path: str) -> List[AppendixRecord]:
    """Parses all appendix records from an HK container."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"HK file not found: {file_path}")

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    if len(file_bytes) < HEADER_SIZE:
        raise ValueError("File is smaller than HK header size")

    magic = file_bytes[0:4]
    if magic != b"HKNT":
        raise ValueError(f"Invalid HK magic: {magic}")

    appendix_offset = struct.unpack_from("<Q", file_bytes, 72)[0]
    if appendix_offset == 0 or appendix_offset >= len(file_bytes):
        return []

    records = []
    offset = appendix_offset

    while offset + REC_HEADER_SIZE <= len(file_bytes):
        hdr_tuple = struct.unpack_from(REC_HEADER_FORMAT, file_bytes, offset)
        (entry_type_raw, flags, name_len, gen, ts, parent_hash,
         loss, acc, pass_rate, custom, target_len, reserved, crc32, data_size) = hdr_tuple
        offset += REC_HEADER_SIZE

        name = file_bytes[offset : offset + name_len].decode("utf-8", errors="replace")
        offset += name_len

        target = file_bytes[offset : offset + target_len].decode("utf-8", errors="replace")
        offset += target_len

        payload = file_bytes[offset : offset + data_size]
        offset += data_size

        # Align to 8 bytes
        if offset % 8 != 0:
            offset += (8 - (offset % 8))

        records.append(AppendixRecord(
            entry_type=AppendixEntryType(entry_type_raw),
            name=name,
            target=target,
            flags=flags,
            generation=gen,
            timestamp=ts,
            parent_hash=parent_hash,
            metrics=AppendixMetrics(loss=loss, accuracy=acc, pass_rate=pass_rate, custom=custom),
            data=payload
        ))

    return records

def append_record(file_path: str, record: AppendixRecord) -> None:
    """Appends an evolutionary record directly to an HK container."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"HK file not found: {file_path}")

    with open(file_path, "rb") as f:
        file_bytes = bytearray(f.read())

    if len(file_bytes) < HEADER_SIZE:
        raise ValueError("File is smaller than HK header size")

    magic = file_bytes[0:4]
    if magic != b"HKNT":
        raise ValueError(f"Invalid HK magic: {magic}")

    flags = struct.unpack_from("<I", file_bytes, 8)[0]
    appendix_offset = struct.unpack_from("<Q", file_bytes, 72)[0]

    # If first appendix record, position at end of file (8-byte aligned)
    file_size = len(file_bytes)
    if appendix_offset == 0:
        aligned_append = (file_size + 7) & ~7
        if aligned_append > file_size:
            file_bytes.extend(b"\x00" * (aligned_append - file_size))
        appendix_offset = len(file_bytes)
        struct.pack_into("<Q", file_bytes, 72, appendix_offset)
        flags |= 0x02  # HAS_APPENDIX
        struct.pack_into("<I", file_bytes, 8, flags)
    else:
        # Align end of existing file to 8 bytes
        aligned_end = (file_size + 7) & ~7
        if aligned_end > file_size:
            file_bytes.extend(b"\x00" * (aligned_end - file_size))

    name_bytes = record.name.encode("utf-8")
    target_bytes = record.target.encode("utf-8")
    ts = record.timestamp if record.timestamp > 0 else int(time.time())

    hdr = struct.pack(
        REC_HEADER_FORMAT,
        int(record.entry_type),
        record.flags,
        len(name_bytes),
        record.generation,
        ts,
        record.parent_hash if len(record.parent_hash) == 32 else b"\x00" * 32,
        record.metrics.loss,
        record.metrics.accuracy,
        record.metrics.pass_rate,
        record.metrics.custom,
        len(target_bytes),
        0,
        0,
        len(record.data),
    )

    file_bytes.extend(hdr)
    file_bytes.extend(name_bytes)
    file_bytes.extend(target_bytes)
    file_bytes.extend(record.data)

    # 8-byte padding
    rem = len(file_bytes) % 8
    if rem != 0:
        file_bytes.extend(b"\x00" * (8 - rem))

    with open(file_path, "wb") as f:
        f.write(file_bytes)

write_appendix_record = append_record

def rollback_appendix(file_path: str, target_generation: int) -> None:
    """Rolls back the HK container to target_generation."""
    records = read_appendix(file_path)
    with open(file_path, "rb") as f:
        file_bytes = bytearray(f.read())

    appendix_offset = struct.unpack_from("<Q", file_bytes, 72)[0]
    if appendix_offset == 0:
        return

    if target_generation <= 0:
        # Clear all appendix entries
        struct.pack_into("<Q", file_bytes, 72, 0)
        flags = struct.unpack_from("<I", file_bytes, 8)[0]
        flags &= ~0x02
        struct.pack_into("<I", file_bytes, 8, flags)
        with open(file_path, "wb") as f:
            f.write(file_bytes[:appendix_offset])
        return

    # Find truncation boundary
    offset = appendix_offset
    truncate_pos = offset
    for r in records:
        if r.generation <= target_generation:
            rec_len = REC_HEADER_SIZE + len(r.name.encode("utf-8")) + len(r.target.encode("utf-8")) + len(r.data)
            rec_len_aligned = (rec_len + 7) & ~7
            offset += rec_len_aligned
            truncate_pos = offset
        else:
            break

    with open(file_path, "wb") as f:
        f.write(file_bytes[:truncate_pos])

def verify_lineage(records_or_path: Union[List[AppendixRecord], str]) -> bool:
    """Cryptographically validates SHA-256 parent hash chaining across generations."""
    if isinstance(records_or_path, str):
        records = read_appendix(records_or_path)
    else:
        records = records_or_path

    if len(records) <= 1:
        return True
    for i in range(1, len(records)):
        prev = records[i - 1]
        curr = records[i]
        expected_full = compute_parent_hash(prev.data, prev.name, prev.target)
        expected_payload = compute_parent_hash(prev.data)
        if curr.parent_hash != expected_full and curr.parent_hash != expected_payload:
            return False
    return True


class AppendixManager:
    """High-level object interface for managing container appendix evolution."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def get_records(self) -> List[AppendixRecord]:
        return read_appendix(self.file_path)

    def append_lora_checkpoint(
        self,
        name: str,
        target: str,
        generation: int,
        adapter_bytes: bytes,
        metrics: Optional[Dict[str, float]] = None,
    ):
        m = AppendixMetrics(
            loss=metrics.get("loss", 0.0) if metrics else 0.0,
            accuracy=metrics.get("accuracy", 0.0) if metrics else 0.0,
            pass_rate=metrics.get("pass_rate", 0.0) if metrics else 0.0,
            custom=metrics.get("custom", 0.0) if metrics else 0.0,
        )
        existing_records = self.get_records()
        parent_hash = b"\x00" * 32
        if existing_records:
            prev = existing_records[-1]
            parent_hash = compute_parent_hash(prev.data, prev.name, prev.target)

        rec = AppendixRecord(
            entry_type=AppendixEntryType.LORA_ADAPTER,
            name=name,
            generation=generation,
            data=adapter_bytes,
            target=target,
            metrics=m,
            parent_hash=parent_hash,
        )
        append_record(self.file_path, rec)

    def rollback(self, target_generation: int):
        rollback_appendix(self.file_path, target_generation)

    def verify(self) -> bool:
        return verify_lineage(self.get_records())

