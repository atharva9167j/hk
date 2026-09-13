#!/usr/bin/env python3
"""
HK Neural Tensor Framework - Unified Python CLI Entrypoint.
Provides command-line utilities for model inspection, GGUF transcoding,
exporting, and GUI launching for pip-installed environments.
"""

import os
import sys
import argparse
import time
from pathlib import Path

import hk
from hk import __version__
from hk.gguf_parser import convert_gguf_to_hk, export_hk_to_gguf, GGUFReaderLight
from hk.torch import safe_open


def cmd_convert_gguf(args):
    in_path = args.input
    out_path = args.output
    if not os.path.exists(in_path):
        print(f"Error: Input file '{in_path}' does not exist.")
        sys.exit(1)

    in_size_mb = os.path.getsize(in_path) / (1024 * 1024)
    print(f"[HK CLI] Ingesting GGUF container: {in_path} ({in_size_mb:.2f} MB)")
    t0 = time.perf_counter()
    convert_gguf_to_hk(in_path, out_path)
    elapsed = (time.perf_counter() - t0) * 1000.0
    out_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[HK CLI] Zero-copy transplant complete in {elapsed:.2f} ms -> {out_path} ({out_size_mb:.2f} MB)")


def cmd_export(args):
    in_path = args.input
    out_path = args.output
    fmt = args.format.lower()
    if not os.path.exists(in_path):
        print(f"Error: Input file '{in_path}' does not exist.")
        sys.exit(1)

    print(f"[HK CLI] Exporting HK container '{in_path}' to format '{fmt}' -> {out_path}")
    t0 = time.perf_counter()
    if fmt == "gguf":
        export_hk_to_gguf(in_path, out_path)
    elif fmt == "safetensors":
        from safetensors.torch import save_file as st_save
        with safe_open(in_path, framework="pt") as reader:
            tensors = {k: reader.get_tensor(k) for k in reader.keys()}
        st_save(tensors, out_path)
    else:
        print(f"Error: Unsupported format '{fmt}'. Choose 'gguf' or 'safetensors'.")
        sys.exit(1)

    elapsed = (time.perf_counter() - t0) * 1000.0
    out_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[HK CLI] Exported successfully in {elapsed:.2f} ms ({out_size_mb:.2f} MB)")


def cmd_inspect(args):
    target = args.path
    if not os.path.exists(target):
        print(f"Error: File '{target}' does not exist.")
        sys.exit(1)

    file_size_mb = os.path.getsize(target) / (1024 * 1024)
    with open(target, "rb") as f:
        magic = f.read(4)

    print("=" * 70)
    print(f"  HK Model Inspector: {os.path.basename(target)} ({file_size_mb:.2f} MB)")
    print("=" * 70)

    if magic == b"HK01":
        with safe_open(target, framework="pt") as f:
            meta = f.metadata()
            keys = f.keys()
            print(f"Format:       HK Neural Tensor Container (v1.0)")
            print(f"Tensors:      {len(keys)}")
            print(f"Metadata KVs: {len(meta)}")
            if meta:
                print("\nMetadata:")
                for k, v in list(meta.items())[:20]:
                    v_str = str(v)
                    if len(v_str) > 60:
                        v_str = v_str[:57] + "..."
                    print(f"  - {k}: {v_str}")
                if len(meta) > 20:
                    print(f"  ... and {len(meta) - 20} more keys")

            print("\nTensors:")
            for name in keys[:25]:
                t = f.get_tensor(name)
                print(f"  - {name:<40} shape={str(list(t.shape)):<18} dtype={str(t.dtype).replace('torch.', '')}")
            if len(keys) > 25:
                print(f"  ... and {len(keys) - 25} more tensors")

    elif magic == b"GGUF":
        reader = GGUFReaderLight(target)
        print(f"Format:       GGUF (v{reader.version})")
        print(f"Tensors:      {len(reader.tensors)}")
        print(f"Metadata KVs: {len(reader.metadata)}")
        if reader.metadata:
            print("\nMetadata:")
            for k, v in list(reader.metadata.items())[:20]:
                v_str = str(v)
                if len(v_str) > 60:
                    v_str = v_str[:57] + "..."
                print(f"  - {k}: {v_str}")
            if len(reader.metadata) > 20:
                print(f"  ... and {len(reader.metadata) - 20} more keys")

        print("\nTensors:")
        for name, desc in list(reader.tensors.items())[:25]:
            print(f"  - {name:<40} shape={str(desc.shape):<18} type={desc.ggml_type} ({desc.size / 1024:.1f} KB)")
        if len(reader.tensors) > 25:
            print(f"  ... and {len(reader.tensors) - 25} more tensors")
    else:
        print(f"Unrecognized magic bytes: {magic}. Expected HK01 or GGUF.")


def cmd_gui(args):
    from hk.gui import launch_gui
    launch_gui(args.file)


def main():
    parser = argparse.ArgumentParser(
        prog="hk",
        description=f"HK Neural Tensor Framework CLI v{__version__}",
    )
    parser.add_argument("--version", action="version", version=f"hk {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    # convert-gguf
    p_conv = subparsers.add_parser("convert-gguf", help="Convert/transcode a GGUF model to HK format")
    p_conv.add_argument("input", help="Path to input .gguf file")
    p_conv.add_argument("output", help="Path to output .hk file")
    p_conv.set_defaults(func=cmd_convert_gguf)

    # export
    p_exp = subparsers.add_parser("export", help="Export HK container to GGUF or Safetensors")
    p_exp.add_argument("input", help="Path to input .hk file")
    p_exp.add_argument("output", help="Path to output file")
    p_exp.add_argument("-f", "--format", default="gguf", choices=["gguf", "safetensors"], help="Target format")
    p_exp.set_defaults(func=cmd_export)

    # inspect
    p_insp = subparsers.add_parser("inspect", help="Inspect container header, metadata, and tensors")
    p_insp.add_argument("path", help="Path to .hk or .gguf file")
    p_insp.set_defaults(func=cmd_inspect)

    # gui
    p_gui = subparsers.add_parser("gui", help="Launch visual HK Model Studio GUI")
    p_gui.add_argument("file", nargs="?", default=None, help="Optional model file to open")
    p_gui.set_defaults(func=cmd_gui)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
