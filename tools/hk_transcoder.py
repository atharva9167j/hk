#!/usr/bin/env python3
"""
HK Container Transcoder & Bidirectional Exporter
Transcode .gguf models to .hk containers with zero-copy bitstream transplant,
and export .hk models to .gguf or .safetensors.
"""

import sys
import os
import time
import argparse

import hk
from hk.gguf_parser import GGUFReaderLight, convert_gguf_to_hk, export_hk_to_gguf


def cmd_convert(args):
    in_path = args.input
    out_path = args.output
    if not os.path.exists(in_path):
        print(f"Error: Input file '{in_path}' does not exist.")
        sys.exit(1)

    print(f"[HK Transcoder] Ingesting GGUF container: {in_path} ({os.path.getsize(in_path) / (1024*1024):.2f} MB)")
    t0 = time.perf_counter()
    convert_gguf_to_hk(in_path, out_path)
    elapsed = (time.perf_counter() - t0) * 1000.0
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[HK Transcoder] Successfully transplanted to HK container in {elapsed:.2f} ms -> {out_path} ({size_mb:.2f} MB)")


def cmd_export(args):
    in_path = args.input
    out_path = args.output
    fmt = args.format.lower()

    if not os.path.exists(in_path):
        print(f"Error: Input file '{in_path}' does not exist.")
        sys.exit(1)

    print(f"[HK Transcoder] Exporting HK model: {in_path} to format '{fmt}'")
    t0 = time.perf_counter()

    if fmt == "gguf":
        export_hk_to_gguf(in_path, out_path)
    elif fmt == "safetensors":
        from safetensors.torch import save_file
        reader = hk.safe_open(in_path, framework="pt")
        tensors = {}
        for k in reader.keys():
            tensors[k] = reader.get_tensor(k)
        save_file(tensors, out_path)
        del reader
    else:
        print(f"Error: Unsupported format '{fmt}'. Supported: gguf, safetensors")
        sys.exit(1)

    elapsed = (time.perf_counter() - t0) * 1000.0
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[HK Transcoder] Successfully exported to {out_path} in {elapsed:.2f} ms ({size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="HK Neural Tensor Transcoder")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Convert GGUF -> HK
    p_convert = subparsers.add_parser("convert", help="Convert GGUF model to HK container")
    p_convert.add_argument("input", help="Path to input .gguf file")
    p_convert.add_argument("output", help="Path to output .hk file")
    p_convert.set_defaults(func=cmd_convert)

    # Export HK -> GGUF / Safetensors
    p_export = subparsers.add_parser("export", help="Export HK container to GGUF or Safetensors")
    p_export.add_argument("input", help="Path to input .hk file")
    p_export.add_argument("output", help="Path to output file")
    p_export.add_argument("-f", "--format", default="gguf", choices=["gguf", "safetensors"], help="Export format")
    p_export.set_defaults(func=cmd_export)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
