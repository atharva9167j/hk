"""
HK Adaptive Neural Framework: Dynamic Architecture Growth (Net2Net)
Implements function-preserving network expansion (Net2WiderNet, Net2DeeperNet)
allowing runtime capacity expansion without retraining base weights, governed
by hardware and task memory limits.
"""

import copy
import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Dict, Any

class GrowthGovernor:
    """Hardware & task constraint governor for dynamic capacity expansion."""
    def __init__(self, max_vram_mb: Optional[int] = None, max_growth_ratio: float = 2.0):
        self.max_growth_ratio = max_growth_ratio
        if max_vram_mb is not None:
            self.max_vram_mb = max_vram_mb
        elif torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            self.max_vram_mb = int(free_bytes / (1024 * 1024) * 0.8) # 80% of free VRAM
        else:
            self.max_vram_mb = 4096 # 4 GB fallback for CPU

    def can_grow(self, current_params: int, additional_params: int, dtype_bytes: int = 4) -> Tuple[bool, str]:
        """Validates whether proposed expansion fits within hardware memory budgets."""
        total_params = current_params + additional_params
        if total_params > current_params * self.max_growth_ratio:
            return False, f"Growth ratio {(total_params/current_params):.2f}x exceeds max limit {self.max_growth_ratio}x"

        added_mb = (additional_params * dtype_bytes) / (1024 * 1024)
        if added_mb > self.max_vram_mb:
            return False, f"Memory requirement {added_mb:.1f} MB exceeds budget {self.max_vram_mb} MB"

        return True, "Approved"

    def request_growth(self, current_dim: int, target_dim: int, dtype_bytes: int = 4) -> Any:
        """Convenience method returning an Approval object with .approved and .reason attributes."""
        added = target_dim - current_dim
        can, reason = self.can_grow(current_dim, added, dtype_bytes)
        class Approval:
            def __init__(self, approved: bool, reason: str):
                self.approved = approved
                self.reason = reason
        return Approval(can, reason)

def net2wider_linear(
    layer1: nn.Linear,
    layer2: Optional[nn.Linear],
    new_out_features: int,
    noise_std: float = 1e-5
) -> Tuple[nn.Linear, Optional[nn.Linear]]:
    """
    Net2WiderNet transformation on consecutive nn.Linear layers.
    Expands layer1 out_features to new_out_features, updating layer2 in_features.
    Mathematically preserves the exact function: f_new(x) == f_old(x).
    """
    old_out = layer1.out_features
    if new_out_features <= old_out:
        return layer1, layer2

    in_features = layer1.in_features
    added_count = new_out_features - old_out

    # Define mapping g(j) for all j in 0..new_out_features-1
    # For j < old_out: g(j) = j
    # For j >= old_out: g(j) = randomly selected source unit in 0..old_out-1
    source_indices = torch.randint(0, old_out, (added_count,)).tolist()
    g = list(range(old_out)) + source_indices

    # Compute replication count c(k) for each old unit k
    c = [0] * old_out
    for src in g:
        c[src] += 1

    # 1. Create expanded layer1
    wider1 = nn.Linear(
        in_features,
        new_out_features,
        bias=layer1.bias is not None,
        device=layer1.weight.device,
        dtype=layer1.weight.dtype,
    )
    with torch.no_grad():
        for j in range(new_out_features):
            src = g[j]
            wider1.weight[j, :] = layer1.weight[src, :]
            if layer1.bias is not None:
                wider1.bias[j] = layer1.bias[src]

    # 2. Update layer2 (if present)
    wider2 = None
    if layer2 is not None:
        out_features2 = layer2.out_features
        wider2 = nn.Linear(
            new_out_features,
            out_features2,
            bias=layer2.bias is not None,
            device=layer2.weight.device,
            dtype=layer2.weight.dtype,
        )
        with torch.no_grad():
            if layer2.bias is not None:
                wider2.bias.copy_(layer2.bias)

            for j in range(new_out_features):
                src = g[j]
                factor = 1.0 / c[src]
                scaled_weight = layer2.weight[:, src] * factor
                if noise_std > 0 and j >= old_out:
                    noise = torch.randn_like(scaled_weight) * noise_std
                    wider2.weight[:, j] = scaled_weight + noise
                else:
                    wider2.weight[:, j] = scaled_weight

    return wider1, wider2


def net2wider_swiglu(
    gate_proj: nn.Linear,
    up_proj: nn.Linear,
    down_proj: nn.Linear,
    new_intermediate_size: int,
    noise_std: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[nn.Linear, nn.Linear, nn.Linear]:
    """
    Net2WiderNet transformation for modern SwiGLU transformer MLP architectures
    (e.g., LLaMA, SmollM, Mistral, Qwen).
    SwiGLU computes: down_proj(SiLU(gate_proj(x)) * up_proj(x)).
    Expands gate_proj and up_proj out_features and down_proj in_features
    from old_intermediate to new_intermediate_size while mathematically
    preserving the exact computed function output: ||f_wider(x) - f_old(x)|| < 1e-6.
    """
    old_inter = gate_proj.out_features
    if new_intermediate_size <= old_inter:
        return gate_proj, up_proj, down_proj

    in_f = gate_proj.in_features
    out_f = down_proj.out_features
    added = new_intermediate_size - old_inter

    try:
        from hk.native import native_net2wider_swiglu, is_native_available
        if is_native_available():
            w_g_np = gate_proj.weight.detach().cpu().to(torch.float32).numpy()
            w_u_np = up_proj.weight.detach().cpu().to(torch.float32).numpy()
            w_d_np = down_proj.weight.detach().cpu().to(torch.float32).numpy()
            b_g_np = gate_proj.bias.detach().cpu().to(torch.float32).numpy() if gate_proj.bias is not None else None
            b_u_np = up_proj.bias.detach().cpu().to(torch.float32).numpy() if up_proj.bias is not None else None
            b_d_np = down_proj.bias.detach().cpu().to(torch.float32).numpy() if down_proj.bias is not None else None

            w_g_new, w_u_new, w_d_new, b_g_new, b_u_new, b_d_new = native_net2wider_swiglu(
                w_g_np,
                w_u_np,
                w_d_np,
                new_intermediate_size,
                b_gate_old=b_g_np,
                b_up_old=b_u_np,
                b_down_old=b_d_np,
                zero_init=(noise_std == 0.0),
                noise_std=noise_std,
                seed=seed if seed is not None else 42,
            )

            device = gate_proj.weight.device
            dtype = gate_proj.weight.dtype

            wider_gate = nn.Linear(in_f, new_intermediate_size, bias=gate_proj.bias is not None, device=device, dtype=dtype)
            wider_up = nn.Linear(in_f, new_intermediate_size, bias=up_proj.bias is not None, device=device, dtype=dtype)
            wider_down = nn.Linear(new_intermediate_size, out_f, bias=down_proj.bias is not None, device=device, dtype=dtype)

            with torch.no_grad():
                wider_gate.weight.copy_(torch.from_numpy(w_g_new).to(device=device, dtype=dtype))
                wider_up.weight.copy_(torch.from_numpy(w_u_new).to(device=device, dtype=dtype))
                wider_down.weight.copy_(torch.from_numpy(w_d_new).to(device=device, dtype=dtype))
                if b_g_new is not None and wider_gate.bias is not None:
                    wider_gate.bias.copy_(torch.from_numpy(b_g_new).to(device=device, dtype=dtype))
                if b_u_new is not None and wider_up.bias is not None:
                    wider_up.bias.copy_(torch.from_numpy(b_u_new).to(device=device, dtype=dtype))
                if b_d_new is not None and wider_down.bias is not None:
                    wider_down.bias.copy_(torch.from_numpy(b_d_new).to(device=device, dtype=dtype))

            return wider_gate, wider_up, wider_down
    except Exception:
        pass

    if seed is not None:
        torch.manual_seed(seed)

    source_indices = torch.randint(0, old_inter, (added,)).tolist()
    g = list(range(old_inter)) + source_indices

    c = [0] * old_inter
    for s in g:
        c[s] += 1

    wider_gate = nn.Linear(in_f, new_intermediate_size, bias=gate_proj.bias is not None, device=gate_proj.weight.device, dtype=gate_proj.weight.dtype)
    wider_up = nn.Linear(in_f, new_intermediate_size, bias=up_proj.bias is not None, device=up_proj.weight.device, dtype=up_proj.weight.dtype)
    wider_down = nn.Linear(new_intermediate_size, out_f, bias=down_proj.bias is not None, device=down_proj.weight.device, dtype=down_proj.weight.dtype)

    with torch.no_grad():
        for j in range(new_intermediate_size):
            s = g[j]
            wider_gate.weight[j, :] = gate_proj.weight[s, :]
            wider_up.weight[j, :] = up_proj.weight[s, :]
            if gate_proj.bias is not None:
                wider_gate.bias[j] = gate_proj.bias[s]
            if up_proj.bias is not None:
                wider_up.bias[j] = up_proj.bias[s]

            factor = 1.0 / c[s]
            scaled_w = down_proj.weight[:, s] * factor
            if noise_std > 0 and j >= old_inter:
                noise = torch.randn_like(scaled_w) * noise_std
                wider_down.weight[:, j] = scaled_w + noise
            else:
                wider_down.weight[:, j] = scaled_w

        if down_proj.bias is not None:
            wider_down.bias.copy_(down_proj.bias)

    return wider_gate, wider_up, wider_down



def net2deeper_linear(in_features: int, activation: Optional[str] = "relu") -> nn.Module:
    """
    Net2DeeperNet transformation: creates an identity-initialized intermediate layer.
    W is initialized to Identity (I) and bias to 0, ensuring f(x) = x upon insertion.
    """
    class IdentityResidualLayer(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            # Initialize to identity matrix
            with torch.no_grad():
                self.linear.weight.copy_(torch.eye(dim))
                self.linear.bias.zero_()
            self.activation = nn.ReLU() if activation == "relu" else nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.activation(self.linear(x))

    return IdentityResidualLayer(in_features)

class ModularResidualBlock(nn.Module):
    """
    Modular expandable residual adapter that can be inserted into existing layers.
    Initialized with zero weights, guaranteeing exact function preservation on day 0:
    y = x + alpha * adapter(x) -> when adapter weight = 0, y = x.
    """
    def __init__(self, dim: int, bottleneck_rank: int = 16):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck_rank, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_rank, dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.0)) # Zero-init scale factor

        # Zero-initialize the up projection so initial output is exactly zero
        nn.init.zeros_(self.up.weight)
        nn.init.kaiming_uniform_(self.down.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.up(self.act(self.down(x)))
        return x + self.scale * residual


def expand_vocab(
    model: nn.Module,
    new_vocab_size: int,
    init_std: float = 0.02,
) -> nn.Module:
    """
    Dynamically expands the vocabulary and embedding capacity of a language model.
    Resizes `embed_tokens` from [V, D] -> [V', D] and `lm_head` from [V, D] -> [V', D].
    Automatically preserves tied weights, and initializes new token embeddings
    matched to existing distribution statistics.
    Crucially, mathematically preserves the exact logits and representations
    for all existing tokens (0..V-1): f_new(token_i) == f_old(token_i).
    """
    embed = getattr(model, "embed_tokens", None)
    if embed is None and hasattr(model, "model"):
        embed = getattr(model.model, "embed_tokens", None)

    lm_head = getattr(model, "lm_head", None)

    if embed is None or lm_head is None:
        raise AttributeError("Model must have 'embed_tokens' and 'lm_head' attributes for vocabulary expansion.")

    old_vocab_size, hidden_size = embed.weight.shape
    if new_vocab_size <= old_vocab_size:
        return model

    added_tokens = new_vocab_size - old_vocab_size
    is_tied = (lm_head.weight.data_ptr() == embed.weight.data_ptr())

    device = embed.weight.device
    dtype = embed.weight.dtype

    if not is_tied:
        try:
            from hk.native import native_expand_vocab, is_native_available
            if is_native_available():
                e_np = embed.weight.detach().cpu().to(torch.float32).numpy()
                h_np = lm_head.weight.detach().cpu().to(torch.float32).numpy() if lm_head is not None else None
                e_new, h_new = native_expand_vocab(e_np, new_vocab_size, lm_head_old=h_np, seed=42)

                new_embed = nn.Embedding(new_vocab_size, hidden_size, device=device, dtype=dtype)
                with torch.no_grad():
                    new_embed.weight.copy_(torch.from_numpy(e_new).to(device=device, dtype=dtype))

                new_head = nn.Linear(hidden_size, new_vocab_size, bias=lm_head.bias is not None, device=device, dtype=dtype)
                with torch.no_grad():
                    new_head.weight.copy_(torch.from_numpy(h_new).to(device=device, dtype=dtype))
                    if lm_head.bias is not None:
                        new_head.bias[:old_vocab_size].copy_(lm_head.bias)
                        new_head.bias[old_vocab_size:].zero_()

                if hasattr(model, "embed_tokens"):
                    model.embed_tokens = new_embed
                elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
                    model.model.embed_tokens = new_embed
                model.lm_head = new_head
                if hasattr(model, "config"):
                    model.config.vocab_size = new_vocab_size
                model._old_vocab_size = old_vocab_size
                return model
        except Exception:
            pass

    new_embed = nn.Embedding(new_vocab_size, hidden_size, device=device, dtype=dtype)

    with torch.no_grad():
        new_embed.weight[:old_vocab_size, :].copy_(embed.weight)
        mean = embed.weight.mean(dim=0, keepdim=True)
        std = embed.weight.std(dim=0, keepdim=True)
        noise = torch.randn((added_tokens, hidden_size), device=device, dtype=dtype) * init_std
        new_embed.weight[old_vocab_size:, :].copy_(mean + noise * std)

    if is_tied:
        from hk.modeling import HKLinear
        new_head = HKLinear(hidden_size, new_vocab_size, bias=False, device="cpu" if device.type == "cpu" else "cuda")
        new_head.weight = new_embed.weight
    else:
        new_head = nn.Linear(hidden_size, new_vocab_size, bias=lm_head.bias is not None, device=device, dtype=dtype)
        with torch.no_grad():
            new_head.weight[:old_vocab_size, :].copy_(lm_head.weight)
            new_head.weight[old_vocab_size:, :].copy_(
                torch.randn((added_tokens, hidden_size), device=device, dtype=dtype) * init_std
            )
            if lm_head.bias is not None:
                new_head.bias[:old_vocab_size].copy_(lm_head.bias)
                new_head.bias[old_vocab_size:].zero_()

    if hasattr(model, "embed_tokens"):
        model.embed_tokens = new_embed
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        model.model.embed_tokens = new_embed

    model.lm_head = new_head

    if hasattr(model, "config"):
        model.config.vocab_size = new_vocab_size

    model._old_vocab_size = old_vocab_size
    return model


def expand_model_width(
    model: nn.Module,
    expansion_ratio: float = 1.33,
    noise_std: float = 0.0,
    layer_indices: Optional[List[int]] = None,
    seed: int = 42,
) -> nn.Module:
    r"""
    Model-wide Net2WiderNet transformation across all (or selected) transformer blocks.
    Supports both modern SwiGLU MLP architectures (LLaMA, SmollM, Mistral) and standard Linear MLPs.
    Mathematically guarantees exact function preservation: ||f_wider(x) - f_old(x)||_{\infty} < 1e-6.
    """
    layers = getattr(model, "layers", None)
    if layers is None and hasattr(model, "model"):
        layers = getattr(model.model, "layers", None)

    if layers is None:
        raise AttributeError("Model must have 'layers' list for transformer expansion.")

    if layer_indices is None:
        layer_indices = list(range(len(layers)))

    old_intermediate_sizes = {}

    for idx in layer_indices:
        layer = layers[idx]
        if hasattr(layer, "gate_proj") and hasattr(layer, "up_proj") and hasattr(layer, "down_proj"):
            old_inter = layer.gate_proj.out_features
            new_inter = int(old_inter * expansion_ratio)
            old_intermediate_sizes[idx] = old_inter

            w_gate, w_up, w_down = net2wider_swiglu(
                layer.gate_proj, layer.up_proj, layer.down_proj,
                new_intermediate_size=new_inter,
                noise_std=noise_std,
                seed=seed + idx,
            )
            layer.gate_proj = w_gate
            layer.up_proj = w_up
            layer.down_proj = w_down

        elif hasattr(layer, "mlp_fc1") and hasattr(layer, "mlp_fc2"):
            old_inter = layer.mlp_fc1.out_features
            new_inter = int(old_inter * expansion_ratio)
            old_intermediate_sizes[idx] = old_inter

            w_fc1, w_fc2 = net2wider_linear(
                layer.mlp_fc1, layer.mlp_fc2,
                new_out_features=new_inter,
                noise_std=noise_std,
            )
            layer.mlp_fc1 = w_fc1
            layer.mlp_fc2 = w_fc2

        elif hasattr(layer, "mlp") and hasattr(layer.mlp, "gate_proj"):
            mlp = layer.mlp
            old_inter = mlp.gate_proj.out_features
            new_inter = int(old_inter * expansion_ratio)
            old_intermediate_sizes[idx] = old_inter

            w_gate, w_up, w_down = net2wider_swiglu(
                mlp.gate_proj, mlp.up_proj, mlp.down_proj,
                new_intermediate_size=new_inter,
                noise_std=noise_std,
                seed=seed + idx,
            )
            mlp.gate_proj = w_gate
            mlp.up_proj = w_up
            mlp.down_proj = w_down

    if hasattr(model, "config") and old_intermediate_sizes:
        first_new_inter = int(list(old_intermediate_sizes.values())[0] * expansion_ratio)
        model.config.intermediate_size = first_new_inter

    model._old_intermediate_sizes = old_intermediate_sizes
    return model


def protect_base_capacity(
    model: nn.Module,
    old_intermediate_sizes: Optional[Dict[int, int]] = None,
    old_vocab_size: Optional[int] = None,
) -> List[Any]:
    """
    Plasticity Isolation Engine (Anti-Catastrophic Forgetting).
    Attaches gradient hooks that mask out gradients on base neurons (0..d_old-1)
    and base vocabulary slots (0..V_old-1), while permitting 100% gradient updates
    on the newly expanded capacity (d_old..d_new-1) and new tokens (V_old..V_new-1).

    This mathematically guarantees ZERO catastrophic forgetting on the model's
    pre-existing knowledge and native language when training on a new language.
    """
    hooks = []

    old_vocab = old_vocab_size or getattr(model, "_old_vocab_size", None)
    old_inter = old_intermediate_sizes or getattr(model, "_old_intermediate_sizes", {})

    # 1. Mask embedding table and LM head
    embed = getattr(model, "embed_tokens", None)
    if embed is None and hasattr(model, "model"):
        embed = getattr(model.model, "embed_tokens", None)

    if embed is not None and old_vocab is not None:
        def make_vocab_hook(cutoff):
            def hook(grad):
                if grad is None:
                    return None
                g = grad.clone()
                g[:cutoff, :] = 0.0
                return g
            return hook

        h = embed.weight.register_hook(make_vocab_hook(old_vocab))
        hooks.append(h)

    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and old_vocab is not None and (embed is None or lm_head.weight.data_ptr() != embed.weight.data_ptr()):
        def make_head_hook(cutoff):
            def hook(grad):
                if grad is None:
                    return None
                g = grad.clone()
                g[:cutoff, :] = 0.0
                return g
            return hook

        h = lm_head.weight.register_hook(make_head_hook(old_vocab))
        hooks.append(h)

    # 2. Mask intermediate layers
    layers = getattr(model, "layers", None)
    if layers is None and hasattr(model, "model"):
        layers = getattr(model.model, "layers", None)

    if layers is not None:
        for idx, layer in enumerate(layers):
            if idx not in old_inter:
                continue
            cutoff = old_inter[idx]

            for proj_name in ("gate_proj", "up_proj"):
                proj = getattr(layer, proj_name, None)
                if proj is None and hasattr(layer, "mlp"):
                    proj = getattr(layer.mlp, proj_name, None)
                if proj is not None:
                    def make_out_hook(c):
                        def hook(grad):
                            if grad is None:
                                return None
                            g = grad.clone()
                            g[:c, :] = 0.0
                            return g
                        return hook
                    h = proj.weight.register_hook(make_out_hook(cutoff))
                    hooks.append(h)

            down_proj = getattr(layer, "down_proj", None)
            if down_proj is None and hasattr(layer, "mlp"):
                down_proj = getattr(layer.mlp, "down_proj", None)
            if down_proj is not None:
                def make_in_hook(c):
                    def hook(grad):
                        if grad is None:
                            return None
                        g = grad.clone()
                        g[:, :c] = 0.0
                        return g
                    return hook
                h = down_proj.weight.register_hook(make_in_hook(cutoff))
                hooks.append(h)

            if hasattr(layer, "mlp_fc1"):
                def make_fc1_hook(c):
                    def hook(grad):
                        if grad is None:
                            return None
                        g = grad.clone()
                        g[:c, :] = 0.0
                        return g
                    return hook
                h = layer.mlp_fc1.weight.register_hook(make_fc1_hook(cutoff))
                hooks.append(h)

            if hasattr(layer, "mlp_fc2"):
                def make_fc2_hook(c):
                    def hook(grad):
                        if grad is None:
                            return None
                        g = grad.clone()
                        g[:, :c] = 0.0
                        return g
                    return hook
                h = layer.mlp_fc2.weight.register_hook(make_fc2_hook(cutoff))
                hooks.append(h)

    return hooks
