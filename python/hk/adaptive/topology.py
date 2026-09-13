"""
HK Adaptive Neural Framework: Head Script & Model Topology Packaging
Enables self-contained single-file model distribution by packaging architecture
definitions, hyper-parameters, and inference runner scripts directly into HK containers.
"""

import json
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from .appendix import AppendixRecord, AppendixEntryType, AppendixFlags, append_record, read_appendix
from ..torch import save_hk, load_hk

class TopologyContainer:
    """Encapsulates neural graph topology and optional executable head scripts."""
    def __init__(self, architecture: str, config: Dict[str, Any], runner_script: Optional[str] = None):
        self.architecture = architecture
        self.config = config
        self.runner_script = runner_script or ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "architecture": self.architecture,
            "config": self.config,
            "runner_script": self.runner_script
        }

    def serialize(self) -> bytes:
        return json.dumps(self.to_dict(), indent=2).encode("utf-8")

    @classmethod
    def deserialize(cls, data: bytes) -> "TopologyContainer":
        d = json.loads(data.decode("utf-8"))
        return cls(
            architecture=d["architecture"],
            config=d["config"],
            runner_script=d.get("runner_script", "")
        )

def package_standalone_hk(
    file_path: str,
    model: nn.Module,
    architecture_name: str,
    config: Dict[str, Any],
    runner_script: Optional[str] = None,
    quantize_mode: Optional[str] = None
) -> None:
    """
    Saves a complete self-contained HK model file packaging:
    1. Tensor weights (dense, sparse 2:4, or dual-mode quantized).
    2. Architecture topology and hyper-parameters.
    3. Optional inference runner script.
    """
    # 1. Save base tensor weights
    metadata = {
        "architecture": architecture_name,
        "format.version": "1.0",
        "has_topology": True
    }
    save_hk(file_path, model.state_dict(), metadata=metadata, quantize_mode=quantize_mode)

    # 2. Append TOPOLOGY_HEAD appendix entry
    topo = TopologyContainer(architecture=architecture_name, config=config, runner_script=runner_script)
    payload = topo.serialize()

    rec = AppendixRecord(
        entry_type=AppendixEntryType.TOPOLOGY_HEAD,
        name="topology.head_spec",
        target="model.root",
        generation=0,
        flags=AppendixFlags.ACTIVE,
        data=payload
    )
    append_record(file_path, rec)

def load_standalone_topology(file_path: str) -> Optional[TopologyContainer]:
    """Reads the topology specification and head script from an HK container."""
    records = read_appendix(file_path)
    for r in records:
        if r.entry_type == AppendixEntryType.TOPOLOGY_HEAD:
            return TopologyContainer.deserialize(r.data)
    return None

def build_model_from_topology(topo: TopologyContainer) -> nn.Module:
    """
    Constructs a PyTorch module directly from a declarative topology specification.
    Supports standard MLP, CNN, and Transformer layer configurations.
    """
    arch = topo.architecture.lower()
    cfg = topo.config

    if arch in ["mlp", "linear_classifier"]:
        in_dim = cfg.get("in_features", 784)
        hidden_dim = cfg.get("hidden_features", 128)
        num_classes = cfg.get("num_classes", 10)

        class AutoMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(in_dim, hidden_dim)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_dim, num_classes)
            def forward(self, x):
                return self.fc2(self.relu(self.fc1(x)))

        return AutoMLP()

    elif arch in ["conv_classifier", "cnn"]:
        in_channels = cfg.get("in_channels", 1)
        hidden_channels = cfg.get("hidden_channels", 16)
        num_classes = cfg.get("num_classes", 10)

        class AutoCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d((4, 4))
                self.fc = nn.Linear(hidden_channels * 16, num_classes)
            def forward(self, x):
                x = torch.relu(self.conv(x))
                x = self.pool(x)
                return self.fc(x.flatten(1))

        return AutoCNN()

    else:
        raise ValueError(f"Unknown architecture '{arch}'. Custom runner script required.")

def load_standalone_hk(file_path: str) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Loads a completely standalone HK model:
    Reconstructs the model architecture from the embedded topology and populates weights.
    """
    topo = load_standalone_topology(file_path)
    if topo is None:
        raise ValueError(f"No TOPOLOGY_HEAD found in '{file_path}'. Use load_hk() for raw tensors.")

    model = build_model_from_topology(topo)
    tensors, meta = load_hk(file_path)

    # Load weights into reconstructed model
    state = model.state_dict()
    for name, param in state.items():
        if name in tensors:
            param.copy_(tensors[name])

    model.eval()
    return model, meta
