"""
HK Graphical Model Editor & Inspector GUI
Cross-platform desktop editor for HK neural tensor format (.hk) containers.
Provides interactive inspection and modification of:
- Container header, flags, and alignment audit
- Full metadata key-value hierarchy with in-place editing
- Tensor table with storage types (K-Quants, IQ, NF4), shapes, sparsity, and memory footprints
- Lineage, training metrics, and appendix historical generations
"""

import sys
import os
import json
import struct
from pathlib import Path
from typing import Optional, Dict, Any, List

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog

# Add parent python package directory to path if needed
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

try:
    import hk
    from hk.format import (
        StorageType, TileLayout, SparsityType,
        STORAGE_F32, STORAGE_F16, STORAGE_BF16,
        STORAGE_Q2_K, STORAGE_Q3_K, STORAGE_Q4_K, STORAGE_Q5_K, STORAGE_Q6_K, STORAGE_Q8_K,
        STORAGE_IQ1_S, STORAGE_IQ4_NL, STORAGE_MXFP4, STORAGE_NVFP4,
    )
    from hk.torch import HKFile, metadata_set
    from hk.native import is_native_available, native_metadata_patch_in_place
except ImportError as e:
    hk = None


STORAGE_TYPE_NAMES = {
    0x00: "F32",
    0x01: "F16",
    0x02: "BF16",
    0x03: "FP8_E4M3",
    0x04: "FP8_E5M2",
    0x05: "INT8",
    0x06: "INT32",
    0x07: "INT64",
    0x08: "UINT8",
    0x09: "BOOL",
    0x10: "DQ4 (NF4)",
    0x11: "DQ8",
    0x12: "DQ6",
    0x13: "DQ12",
    0x14: "DQT (BitNet)",
    0x20: "SPARSE_F16",
    0x21: "SPARSE_DQ8",
    0x22: "SPARSE_2_4",
    0x23: "SPARSE_DQ4_2_4",
    0x30: "NULL_REF",
    0x31: "SHARED_REF",
    0x32: "LORA_REF",
    0x40: "Q2_K",
    0x41: "Q3_K",
    0x42: "Q4_K",
    0x43: "Q5_K",
    0x44: "Q6_K",
    0x45: "Q8_K",
    0x50: "IQ1_S",
    0x51: "IQ1_M",
    0x52: "IQ2_XXS",
    0x53: "IQ2_XS",
    0x54: "IQ3_XXS",
    0x55: "IQ4_NL",
    0x56: "IQ4_XS",
    0x60: "TQ1_0",
    0x61: "TQ2_0",
    0x62: "MXFP4",
    0x63: "NVFP4",
}


class HKEditorApp(tk.Tk):
    def __init__(self, initial_path: Optional[str] = None):
        super().__init__()
        self.title("HK Model Editor & Tensor Inspector")
        self.geometry("1100;750".replace(";", "x"))
        self.minsize(850, 550)

        self.current_file: Optional[Path] = None
        self.header_info: Dict[str, Any] = {}
        self.metadata_dict: Dict[str, Any] = {}
        self.tensors_list: List[Dict[str, Any]] = []
        self.appendix_records: List[Dict[str, Any]] = []

        self._setup_styles()
        self._setup_menu()
        self._setup_ui()

        if initial_path and os.path.isfile(initial_path):
            self.load_model(initial_path)

    def _setup_styles(self):
        self.style = ttk.Style()
        theme_names = self.style.theme_names()
        if "clam" in theme_names:
            self.style.theme_use("clam")

        self.style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))
        self.style.configure("Treeview", font=("Consolas", 9), rowheight=24)
        self.style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"))
        self.style.configure("SubHeader.TLabel", font=("Segoe UI", 10, "bold"))
        self.style.configure("Status.TLabel", font=("Segoe UI", 9))

    def _setup_menu(self):
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Model (.hk)...", command=self.on_open, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Export Metadata to JSON...", command=self.on_export_metadata_json)
        file_menu.add_command(label="Import Metadata from JSON...", command=self.on_import_metadata_json)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="Audit Alignment & 128-Byte Integrity", command=self.on_audit_alignment)
        tools_menu.add_command(label="Show Format Specifications", command=self.on_show_specs)
        menubar.add_cascade(label="Tools", menu=tools_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About HK Format", command=self.on_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)
        self.bind("<Control-o>", lambda e: self.on_open())

    def _setup_ui(self):
        top_frame = ttk.Frame(self, padding=(10, 8, 10, 8))
        top_frame.pack(fill=tk.X)

        self.file_label = ttk.Label(top_frame, text="No model file opened", style="Header.TLabel")
        self.file_label.pack(side=tk.LEFT)

        open_btn = ttk.Button(top_frame, text="Open File...", command=self.on_open)
        open_btn.pack(side=tk.RIGHT, padx=4)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.tab_overview = ttk.Frame(self.notebook, padding=10)
        self.tab_metadata = ttk.Frame(self.notebook, padding=10)
        self.tab_tensors = ttk.Frame(self.notebook, padding=10)
        self.tab_appendix = ttk.Frame(self.notebook, padding=10)

        self.notebook.add(self.tab_overview, text="  Container Overview  ")
        self.notebook.add(self.tab_metadata, text="  Metadata & Hyperparameters  ")
        self.notebook.add(self.tab_tensors, text="  Tensors & Quantization  ")
        self.notebook.add(self.tab_appendix, text="  Lineage & Appendix  ")

        self._build_overview_tab()
        self._build_metadata_tab()
        self._build_tensors_tab()
        self._build_appendix_tab()

        self.status_bar = ttk.Label(
            self, text="Ready. Zero-copy HK Neural Tensor Format Engine.",
            relief=tk.SUNKEN, anchor=tk.W, style="Status.TLabel", padding=(6, 4)
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _build_overview_tab(self):
        frame = self.tab_overview

        hdr_box = ttk.LabelFrame(frame, text="Container Header & Hardware Alignment", padding=10)
        hdr_box.pack(fill=tk.X, pady=(0, 10))

        self.overview_labels: Dict[str, ttk.Label] = {}
        fields = [
            ("magic", "Magic Identifier:"),
            ("version", "Format Version:"),
            ("alignment", "Memory Alignment:"),
            ("endianness", "Container Endianness:"),
            ("flags", "Header Bit Flags:"),
            ("tensors_count", "Total Tensor Count:"),
            ("meta_kv_count", "Metadata Key-Value Count:"),
            ("toc_offset", "Table of Contents Offset:"),
            ("data_offset", "Tensor Data Base Offset:"),
            ("file_size", "Container File Size:"),
        ]

        grid_frame = ttk.Frame(hdr_box)
        grid_frame.pack(fill=tk.X)
        for i, (key, label_text) in enumerate(fields):
            row = i // 2
            col = (i % 2) * 2
            lbl_title = ttk.Label(grid_frame, text=label_text, font=("Segoe UI", 9, "bold"))
            lbl_title.grid(row=row, column=col, sticky=tk.W, padx=(10, 5), pady=3)
            lbl_val = ttk.Label(grid_frame, text="-", font=("Consolas", 9))
            lbl_val.grid(row=row, column=col + 1, sticky=tk.W, padx=(0, 20), pady=3)
            self.overview_labels[key] = lbl_val

        stat_box = ttk.LabelFrame(frame, text="Model Statistics & Quantization Summary", padding=10)
        stat_box.pack(fill=tk.BOTH, expand=True)

        self.stat_text = tk.Text(stat_box, wrap=tk.WORD, font=("Consolas", 10), relief=tk.FLAT)
        self.stat_text.pack(fill=tk.BOTH, expand=True)
        self.stat_text.insert(tk.END, "Open a .hk file to view model architecture and tensor distributions.")
        self.stat_text.config(state=tk.DISABLED)

    def _build_metadata_tab(self):
        frame = self.tab_metadata

        bar = ttk.Frame(frame)
        bar.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(bar, text="Filter:").pack(side=tk.LEFT, padx=(0, 4))
        self.meta_filter_var = tk.StringVar()
        self.meta_filter_var.trace_add("write", lambda *args: self._filter_metadata())
        entry = ttk.Entry(bar, textvariable=self.meta_filter_var, width=30)
        entry.pack(side=tk.LEFT, padx=(0, 10))

        add_btn = ttk.Button(bar, text="Add Key...", command=self.on_add_metadata_key)
        add_btn.pack(side=tk.LEFT, padx=4)

        edit_btn = ttk.Button(bar, text="Edit Selected...", command=self.on_edit_metadata_key)
        edit_btn.pack(side=tk.LEFT, padx=4)

        del_btn = ttk.Button(bar, text="Delete Selected", command=self.on_delete_metadata_key)
        del_btn.pack(side=tk.LEFT, padx=4)

        exp_btn = ttk.Button(bar, text="Export JSON", command=self.on_export_metadata_json)
        exp_btn.pack(side=tk.RIGHT, padx=4)

        columns = ("key", "type", "value")
        self.meta_tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="browse")
        self.meta_tree.heading("key", text="Metadata Key")
        self.meta_tree.heading("type", text="Data Type")
        self.meta_tree.heading("value", text="Value")

        self.meta_tree.column("key", width=340, anchor=tk.W)
        self.meta_tree.column("type", width=120, anchor=tk.CENTER)
        self.meta_tree.column("value", width=550, anchor=tk.W)

        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.meta_tree.yview)
        self.meta_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.meta_tree.pack(fill=tk.BOTH, expand=True)

        self.meta_tree.bind("<Double-1>", lambda e: self.on_edit_metadata_key())

    def _build_tensors_tab(self):
        frame = self.tab_tensors

        bar = ttk.Frame(frame)
        bar.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(bar, text="Search Tensor:").pack(side=tk.LEFT, padx=(0, 4))
        self.tensor_filter_var = tk.StringVar()
        self.tensor_filter_var.trace_add("write", lambda *args: self._filter_tensors())
        entry = ttk.Entry(bar, textvariable=self.tensor_filter_var, width=32)
        entry.pack(side=tk.LEFT, padx=(0, 10))

        self.tensor_count_lbl = ttk.Label(bar, text="0 tensors listed")
        self.tensor_count_lbl.pack(side=tk.LEFT, padx=10)

        columns = ("idx", "name", "storage", "shape", "elements", "size", "offset", "sparsity")
        self.tensor_tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="browse")
        self.tensor_tree.heading("idx", text="#")
        self.tensor_tree.heading("name", text="Tensor Name")
        self.tensor_tree.heading("storage", text="Storage / Quant")
        self.tensor_tree.heading("shape", text="Shape")
        self.tensor_tree.heading("elements", text="Elements")
        self.tensor_tree.heading("size", text="Size (Bytes)")
        self.tensor_tree.heading("offset", text="Offset")
        self.tensor_tree.heading("sparsity", text="Sparsity %")

        self.tensor_tree.column("idx", width=45, anchor=tk.CENTER)
        self.tensor_tree.column("name", width=320, anchor=tk.W)
        self.tensor_tree.column("storage", width=110, anchor=tk.CENTER)
        self.tensor_tree.column("shape", width=150, anchor=tk.CENTER)
        self.tensor_tree.column("elements", width=110, anchor=tk.E)
        self.tensor_tree.column("size", width=110, anchor=tk.E)
        self.tensor_tree.column("offset", width=100, anchor=tk.E)
        self.tensor_tree.column("sparsity", width=90, anchor=tk.CENTER)

        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.tensor_tree.yview)
        self.tensor_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tensor_tree.pack(fill=tk.BOTH, expand=True)

    def _build_appendix_tab(self):
        frame = self.tab_appendix

        top_info = ttk.Label(
            frame,
            text="Checkpoint Lineage & Appendix Log: Tracks training steps, validation metrics, and model versions.",
            style="SubHeader.TLabel"
        )
        top_info.pack(anchor=tk.W, pady=(0, 8))

        columns = ("gen", "timestamp", "name", "loss", "accuracy", "pass_rate", "payload_size")
        self.app_tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="browse")
        self.app_tree.heading("gen", text="Gen #")
        self.app_tree.heading("timestamp", text="Timestamp (UTC)")
        self.app_tree.heading("name", text="Checkpoint Event")
        self.app_tree.heading("loss", text="Loss")
        self.app_tree.heading("accuracy", text="Accuracy")
        self.app_tree.heading("pass_rate", text="Pass Rate")
        self.app_tree.heading("payload_size", text="Payload Size")

        self.app_tree.column("gen", width=60, anchor=tk.CENTER)
        self.app_tree.column("timestamp", width=180, anchor=tk.CENTER)
        self.app_tree.column("name", width=250, anchor=tk.W)
        self.app_tree.column("loss", width=90, anchor=tk.E)
        self.app_tree.column("accuracy", width=90, anchor=tk.E)
        self.app_tree.column("pass_rate", width=90, anchor=tk.E)
        self.app_tree.column("payload_size", width=110, anchor=tk.E)

        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.app_tree.yview)
        self.app_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.app_tree.pack(fill=tk.BOTH, expand=True)

    def on_open(self):
        file_selected = filedialog.askopenfilename(
            title="Open HK Neural Tensor File",
            filetypes=[("HK Model Files (*.hk)", "*.hk"), ("All Files (*.*)", "*.*")]
        )
        if file_selected:
            self.load_model(file_selected)

    def load_model(self, file_path_str: str):
        path = Path(file_path_str)
        if not path.is_file():
            messagebox.showerror("File Error", f"File not found: {file_path_str}")
            return

        try:
            self._read_hk_file(path)
            self.current_file = path
            self.file_label.config(text=f"{path.name}  ({path.parent})")
            self._update_overview_ui()
            self._filter_metadata()
            self._filter_tensors()
            self._update_appendix_ui()
            self.status_bar.config(text=f"Loaded {path.name} ({len(self.tensors_list)} tensors, {len(self.metadata_dict)} metadata entries)")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to parse HK container:\n{str(e)}")

    def _read_hk_file(self, path: Path):
        file_size = path.stat().st_size
        with open(path, "rb") as f:
            header_bytes = f.read(128)
            if len(header_bytes) < 128:
                raise ValueError("File is smaller than 128-byte HK header")

            magic = header_bytes[:4]
            if magic != b"HKNT":
                raise ValueError(f"Invalid HK magic: {magic} (expected b'HKNT')")

            v_maj, v_min, flags, reserved_flags = struct.unpack("<BBHH", header_bytes[4:10])
            alignment = struct.unpack("<I", header_bytes[10:14])[0]
            tensors_count, meta_kv_count = struct.unpack("<QQ", header_bytes[14:30])
            toc_offset, data_offset, appendix_offset = struct.unpack("<QQQ", header_bytes[30:54])

            endianness = "Little-Endian"
            if flags & 0x0080:
                endianness = "Big-Endian"

            self.header_info = {
                "magic": magic.decode("ascii", errors="replace"),
                "version": f"{v_maj}.{v_min}",
                "alignment": f"{alignment} bytes (Strict 128B)" if (flags & 0x0008) else f"{alignment} bytes",
                "endianness": endianness,
                "flags": f"0x{flags:04X} (mmap_cow={bool(flags & 1)}, sharded={bool(flags & 0x40)}, appendix={bool(flags & 2)})",
                "tensors_count": str(tensors_count),
                "meta_kv_count": str(meta_kv_count),
                "toc_offset": f"0x{toc_offset:X} ({toc_offset} bytes)",
                "data_offset": f"0x{data_offset:X} ({data_offset} bytes)",
                "file_size": f"{file_size:,} bytes ({file_size / (1024*1024):.2f} MB)",
            }

            f.seek(toc_offset)
            self.tensors_list = []
            for i in range(tensors_count):
                name_len = struct.unpack("<H", f.read(2))[0]
                name_bytes = f.read(name_len)
                name = name_bytes.decode("utf-8", errors="replace")

                st_type, tl_layout, sp_type, ndim = struct.unpack("<BBBB", f.read(4))
                shape = []
                for _ in range(8):
                    dim_val = struct.unpack("<Q", f.read(8))[0]
                    shape.append(dim_val)
                shape = shape[:ndim]

                data_off, d_size, res_off, res_size, sc_off, sc_size = struct.unpack("<QQQQQQ", f.read(48))
                blk_size = struct.unpack("<H", f.read(2))[0]
                sp_ratio = struct.unpack("<f", f.read(4))[0]

                elements = 1
                for d in shape:
                    elements *= d

                st_name = STORAGE_TYPE_NAMES.get(st_type, f"0x{st_type:02X}")
                self.tensors_list.append({
                    "index": i,
                    "name": name,
                    "storage_type": st_type,
                    "storage_name": st_name,
                    "tile_layout": tl_layout,
                    "sparsity_type": sp_type,
                    "ndim": ndim,
                    "shape": shape,
                    "elements": elements,
                    "data_offset": data_off,
                    "data_size": d_size,
                    "block_size": blk_size,
                    "sparsity_ratio": sp_ratio,
                })

        self.metadata_dict = {}
        if hk is not None:
            try:
                with HKFile(str(path)) as hf:
                    self.metadata_dict = hf.metadata()
            except Exception:
                pass

        self.appendix_records = []
        if (flags & 0x0002) and appendix_offset > 0:
            try:
                with open(path, "rb") as f:
                    f.seek(appendix_offset)
                    app_magic = f.read(4)
                    if app_magic == b"HKAP":
                        app_ver, num_records = struct.unpack("<HI", f.read(6))
                        for gen in range(num_records):
                            rec_len = struct.unpack("<I", f.read(4))[0]
                            rec_bytes = f.read(rec_len)
                            ts, r_gen, loss, acc, p_rate = struct.unpack("<Q I f f f", rec_bytes[:24])
                            self.appendix_records.append({
                                "gen": r_gen,
                                "timestamp": str(ts),
                                "name": f"Generation {r_gen} Checkpoint",
                                "loss": f"{loss:.4f}",
                                "accuracy": f"{acc:.4f}",
                                "pass_rate": f"{p_rate:.4f}",
                                "payload_size": f"{rec_len} bytes",
                            })
            except Exception:
                pass

    def _update_overview_ui(self):
        for k, v in self.header_info.items():
            if k in self.overview_labels:
                self.overview_labels[k].config(text=v)

        total_elements = sum(t["elements"] for t in self.tensors_list)
        total_data_bytes = sum(t["data_size"] for t in self.tensors_list)

        quant_counts: Dict[str, int] = {}
        for t in self.tensors_list:
            sn = t["storage_name"]
            quant_counts[sn] = quant_counts.get(sn, 0) + 1

        summary_lines = [
            f"Model Architecture Analysis:",
            f"--------------------------------------------------",
            f"Total Tensors      : {len(self.tensors_list):,}",
            f"Total Parameters   : {total_elements:,} elements",
            f"Total Weight Bytes : {total_data_bytes:,} bytes ({total_data_bytes / (1024*1024):.2f} MB)",
            f"",
            f"Quantization Breakdown:",
        ]
        for sn, count in sorted(quant_counts.items(), key=lambda x: -x[1]):
            summary_lines.append(f"  * {sn:<18}: {count:>3} tensors ({count / len(self.tensors_list) * 100:.1f}%)")

        arch = self.metadata_dict.get("general.architecture", "Transformer")
        summary_lines.extend([
            f"",
            f"Model Metadata Taxonomy:",
            f"  * Architecture   : {arch}",
            f"  * Model Name     : {self.metadata_dict.get('general.name', 'Untitled')}",
            f"  * Context Length : {self.metadata_dict.get(f'{arch}.context_length', 'N/A')}",
            f"  * Embedding Dim  : {self.metadata_dict.get(f'{arch}.embedding_length', 'N/A')}",
            f"  * Layers/Blocks  : {self.metadata_dict.get(f'{arch}.block_count', 'N/A')}",
            f"  * Attention Heads: {self.metadata_dict.get(f'{arch}.attention.head_count', 'N/A')}",
        ])

        self.stat_text.config(state=tk.NORMAL)
        self.stat_text.delete("1.0", tk.END)
        self.stat_text.insert(tk.END, "\n".join(summary_lines))
        self.stat_text.config(state=tk.DISABLED)

    def _filter_metadata(self):
        query = self.meta_filter_var.get().lower().strip()
        for row in self.meta_tree.get_children():
            self.meta_tree.delete(row)

        for k, v in sorted(self.metadata_dict.items()):
            if query and query not in k.lower() and query not in str(v).lower():
                continue
            t_name = type(v).__name__
            val_str = str(v)
            if len(val_str) > 200:
                val_str = val_str[:197] + "..."
            self.meta_tree.insert("", tk.END, values=(k, t_name, val_str))

    def _filter_tensors(self):
        query = self.tensor_filter_var.get().lower().strip()
        for row in self.tensor_tree.get_children():
            self.tensor_tree.delete(row)

        matched = 0
        for t in self.tensors_list:
            if query and query not in t["name"].lower() and query not in t["storage_name"].lower():
                continue
            matched += 1
            shape_str = "x".join(str(x) for x in t["shape"])
            self.tensor_tree.insert(
                "", tk.END,
                values=(
                    t["index"],
                    t["name"],
                    t["storage_name"],
                    shape_str,
                    f"{t['elements']:,}",
                    f"{t['data_size']:,}",
                    f"0x{t['data_offset']:X}",
                    f"{t['sparsity_ratio'] * 100:.1f}%",
                )
            )
        self.tensor_count_lbl.config(text=f"{matched} of {len(self.tensors_list)} tensors listed")

    def _update_appendix_ui(self):
        for row in self.app_tree.get_children():
            self.app_tree.delete(row)
        for r in self.appendix_records:
            self.app_tree.insert("", tk.END, values=(
                r["gen"], r["timestamp"], r["name"], r["loss"], r["accuracy"], r["pass_rate"], r["payload_size"]
            ))

    def on_add_metadata_key(self):
        if not self.current_file:
            messagebox.showwarning("Warning", "Please open a model container first.")
            return

        key = simpledialog.askstring("Add Metadata", "Enter Metadata Key (e.g. general.description):", parent=self)
        if not key:
            return
        val = simpledialog.askstring("Add Metadata", f"Enter Value for '{key}':", parent=self)
        if val is None:
            return

        self.metadata_dict[key] = val
        self._filter_metadata()

        if native_metadata_patch_in_place(str(self.current_file), key, val):
            self.status_bar.config(text=f"Patched '{key}' in-place via native Zig engine.")
        else:
            self.status_bar.config(text=f"Added key '{key}' in session memory.")

    def on_edit_metadata_key(self):
        selected = self.meta_tree.selection()
        if not selected:
            messagebox.showinfo("Select Key", "Please select a metadata key to edit.")
            return

        item = self.meta_tree.item(selected[0])
        key = item["values"][0]
        cur_val = self.metadata_dict.get(key, "")

        new_val = simpledialog.askstring("Edit Metadata", f"Modify value for '{key}':", initialvalue=str(cur_val), parent=self)
        if new_val is None:
            return

        self.metadata_dict[key] = new_val
        self._filter_metadata()

        if self.current_file and native_metadata_patch_in_place(str(self.current_file), key, new_val):
            self.status_bar.config(text=f"Updated '{key}' in-place via native Zig engine.")
        else:
            self.status_bar.config(text=f"Updated '{key}' in session memory.")

    def on_delete_metadata_key(self):
        selected = self.meta_tree.selection()
        if not selected:
            return
        item = self.meta_tree.item(selected[0])
        key = item["values"][0]
        if messagebox.askyesno("Confirm Delete", f"Remove metadata key '{key}'?"):
            if key in self.metadata_dict:
                del self.metadata_dict[key]
                self._filter_metadata()
                self.status_bar.config(text=f"Removed '{key}' from session metadata.")

    def on_export_metadata_json(self):
        if not self.metadata_dict:
            messagebox.showwarning("Warning", "No metadata to export.")
            return
        out_path = filedialog.asksaveasfilename(
            title="Export Metadata JSON",
            defaultextension=".json",
            filetypes=[("JSON Files (*.json)", "*.json")]
        )
        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata_dict, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("Export Successful", f"Exported {len(self.metadata_dict)} metadata keys to:\n{out_path}")

    def on_import_metadata_json(self):
        in_path = filedialog.askopenfilename(
            title="Import Metadata JSON",
            filetypes=[("JSON Files (*.json)", "*.json")]
        )
        if in_path and os.path.isfile(in_path):
            with open(in_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.metadata_dict.update(data)
                self._filter_metadata()
                messagebox.showinfo("Import Successful", f"Imported {len(data)} keys from JSON.")

    def on_audit_alignment(self):
        if not self.current_file:
            messagebox.showwarning("Warning", "No file currently loaded.")
            return
        misaligned = []
        for t in self.tensors_list:
            if t["data_offset"] % 128 != 0:
                misaligned.append((t["name"], t["data_offset"]))

        if not misaligned:
            messagebox.showinfo("Alignment Audit Passed", f"All {len(self.tensors_list)} tensors strictly satisfy AVX-512 / Tensor Core 128-byte hardware alignment.")
        else:
            msg = f"Found {len(misaligned)} misaligned tensors:\n"
            for name, off in misaligned[:5]:
                msg += f"  * {name}: offset 0x{off:X} (rem={off % 128})\n"
            messagebox.showwarning("Alignment Audit Warning", msg)

    def on_show_specs(self):
        messagebox.showinfo(
            "HK Format Specification",
            "HK Neural Tensor Format (HKNT) v1.0\n\n"
            "* 128-Byte Strict Hardware Alignment for zero-copy DMA & GPU mmap\n"
            "* Advanced Quantization Zoo: K-Quants (Q2_K..Q8_K), IQ-Quants, NF4, BitNet\n"
            "* 200+ Standardized taxonomy keys for MLA, YaRN, MoE, and SSM\n"
            "* Zero Python/protobuf external dependencies"
        )

    def on_about(self):
        messagebox.showinfo("About HK Model Editor", "HK Graphical Model Editor v1.0.0\nHigh-Performance Neural Format Studio")


def main():
    initial_path = sys.argv[1] if len(sys.argv) > 1 else None
    app = HKEditorApp(initial_path)
    app.mainloop()


if __name__ == "__main__":
    main()
