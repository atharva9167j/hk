# In-Container Version Lineage and Instant Rollback

In this guide, I explain the Appendix region in `.hk` containers, which I designed for cryptographic version tracking and sub-millisecond rollback without duplicating base model weights.

---

## The Problem with Traditional Model Versioning

If you train or fine-tune models today, managing checkpoints is painful:
- Duplicated Checkpoints: Each training step or epoch saves another multi-gigabyte folder. If you save 10 checkpoints of a 7B model, you burn through 140 gigabytes of disk space immediately.
- Git LFS Overhead: Pushing model updates to Git repositories with LFS means cloning and pulling massive binary files repeatedly.
- No Audit Trail: Once weights are overwritten or adjusted, you have no easy cryptographic way to verify which dataset, learning rate, or code generation created that specific state.
- Accidental Poisoning: If a fine-tuning run encounters poisoned data or collapses into repetitive loops, rolling back means finding an old backup or retraining from scratch.

I fixed this in HK by embedding an append-only version Directed Acyclic Graph (DAG) directly inside the `.hk` container: the **Appendix**.

---

## How the Appendix Works

The Appendix resides at the tail end of the `.hk` container (`header.appendix_offset`). Base model weights live untouched at the beginning of the file.

```
+-------------------------------------------------------------+
| Header & TOC                                                |
+-------------------------------------------------------------+
| Base Model Weights (Unchanged, zero-copy mapped)            |
+-------------------------------------------------------------+
| Appendix Region (Append-Only Version DAG)                   |
|   - Generation 1: Initial base model state [Hash: 0x4a8f...] |
|   - Generation 2: LoRA fine-tune on Python [Parent: 0x4a8f] |
|   - Generation 3: Net2WiderNet expansion   [Parent: 0x89b1] |
|   - Generation 4: Self-play training run   [Parent: 0xef20] |
+-------------------------------------------------------------+
```

When you fine-tune with `HKTrainer` or run self-training cycles:
1. Base weights remain untouched.
2. Only the delta changes (e.g. low-rank adapter updates, dimension growth maps, or evaluation metrics) are appended as a new record at the end of the file.
3. Each record contains a cryptographic SHA-256 hash of its parent state, creating an unbreakable chain of custody.

---

## 1. Inspecting Version History

You can view the full lineage of any `.hk` file directly from your terminal:

```bash
hk appendix list model.hk
```

Example output:
```
================================================================================
APPENDIX LINEAGE: model.hk
================================================================================
Gen | Type       | Step  | Loss   | Pass Rate | Timestamp           | Parent Hash
----+------------+-------+--------+-----------+---------------------+------------
  1 | BASE_STATE |     0 | 0.0000 |    0.0%   | 2026-03-01 10:14:00 | [Root]
  2 | LORA_DELTA |  1500 | 1.4210 |   48.5%   | 2026-03-02 14:22:15 | 4a8fe901...
  3 | NET2WIDER  |  1500 | 1.4210 |   48.5%   | 2026-03-02 18:05:30 | 89b144fa...
  4 | SELF_PLAY  |  3000 | 0.8920 |   76.2%   | 2026-03-03 09:40:12 | ef20cc7b...
================================================================================
Total Appendix Records: 4 | Total Overhead: 24.8 MB (vs 28.0 GB duplicated)
```

Notice the storage efficiency: four distinct model generations tracked in a single file with only 24.8 MB of delta overhead instead of duplicating four full 7 GB checkpoints.

---

## 2. Instant Generational Rollback

If Generation 4 suffered from data poisoning or degraded quality on benchmark tests, you can instantly rollback the container to Generation 2:

```bash
hk rollback model.hk --generation 2
```

The rollback executes in under a millisecond. HK resets the active pointer and restores the Table of Contents to the exact state it had at Generation 2. You do not need to download anything or restore from a backup.

---

## 3. Python API (`AppendixManager`)

You can also manage the Appendix programmatically in Python:

```python
from hk.adaptive.appendix import AppendixManager

manager = AppendixManager("model.hk")

# List all generational records
records = manager.list_records()
for r in records:
    print(f"Gen {r.generation} ({r.record_type}): loss={r.metrics.get('loss')}")

# Check cryptographic integrity
is_valid, reason = manager.verify_chain_integrity()
print("Cryptographic Lineage Valid:", is_valid)

# Rollback to generation 2
if is_valid:
    manager.rollback_to_generation(target_generation=2)
    print("Rollback successful. Container active state restored to Gen 2.")
```

---

## Security & Tamper Detection

Because each record links to the cryptographic hash of the entire container state before that record was appended:
- Modifying earlier weights invalidates all downstream records.
- If someone edits a byte in the base model or an earlier adapter, `hk verify` and `manager.verify_chain_integrity()` immediately report a hash mismatch and pinpoint the exact tampered record.
- This provides enterprise-grade provenance and safety for deployed open weights.
