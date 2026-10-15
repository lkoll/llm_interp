from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from torch import nn

# Infimum of each activation function on (-inf, 0) — bounds the MLP-neuron
# "leak_neg"/"double_neg" regimes (Addendum A). Both SiLU and GELU-tanh are
# sign-preserving, so the regime logic carries over; GELU-tanh is unused until
# a Gemma 3 adapter exists.
_ACTIVATION_NEGATIVE_MIN = {
    "silu": -0.2784645427610738,
    "gelu_pytorch_tanh": -0.17000188,
}


class ArchAdapter(ABC):
    """Maps logical sites (embeddings, resid_pre[l], attention input norm, q/k/v
    projections, q/k norms, rotary embedding, final norm, lm_head) to the actual
    modules of a loaded model, plus a handful of architecture-specific recompute
    primitives. One adapter per model family; all dims are read from the model's
    own config, never hardcoded.

    Per-layer methods may return `None` for a site that doesn't exist at that layer
    (e.g. a future architecture with QK-norm on only some layers) — callers must
    handle that.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.config = model.config

    def _check_layer_idx(self, layer_idx: int) -> None:
        if not (0 <= layer_idx < self.num_layers):
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {self.num_layers})")

    # --- dims, read from config ---

    @property
    @abstractmethod
    def num_layers(self) -> int: ...

    @property
    @abstractmethod
    def hidden_size(self) -> int: ...

    @property
    @abstractmethod
    def num_attention_heads(self) -> int: ...

    @property
    @abstractmethod
    def num_key_value_heads(self) -> int: ...

    @property
    @abstractmethod
    def head_dim(self) -> int: ...

    @property
    @abstractmethod
    def intermediate_size(self) -> int:
        """MLP width (`d_mlp`) — Addendum A neuron capture."""

    # --- logical sites: modules ---

    @abstractmethod
    def embeddings(self) -> nn.Module: ...

    @abstractmethod
    def layer(self, layer_idx: int) -> nn.Module:
        """The module whose input is resid_pre[layer_idx] — hook its forward-pre for capture."""

    @abstractmethod
    def attn_input_norm(self, layer_idx: int) -> nn.Module | None: ...

    @abstractmethod
    def q_proj(self, layer_idx: int) -> nn.Module | None: ...

    @abstractmethod
    def k_proj(self, layer_idx: int) -> nn.Module | None: ...

    @abstractmethod
    def v_proj(self, layer_idx: int) -> nn.Module | None: ...

    @abstractmethod
    def q_norm(self, layer_idx: int) -> nn.Module | None: ...

    @abstractmethod
    def k_norm(self, layer_idx: int) -> nn.Module | None: ...

    @abstractmethod
    def rotary_emb(self, layer_idx: int) -> nn.Module | None:
        """The rotary embedding module responsible for this layer (may differ per layer
        for architectures with multiple RoPE bases)."""

    @abstractmethod
    def final_norm(self) -> nn.Module: ...

    @abstractmethod
    def lm_head(self) -> nn.Module: ...

    @abstractmethod
    def attention_type(self, layer_idx: int) -> str:
        """e.g. 'full_attention' or 'sliding_attention', per layer."""

    @abstractmethod
    def attention_scaling(self, layer_idx: int) -> float:
        """The model's own attention score scaling factor for this layer, read from
        the real submodule rather than recomputed as head_dim**-0.5 — some
        architectures (e.g. Gemma 3) use a custom scaling."""

    # --- MLP sites (Addendum A: native neuron capture) ---

    @abstractmethod
    def mlp(self, layer_idx: int) -> nn.Module:
        """The MLP submodule as a whole — hook its forward-pre for `mlp_in`,
        forward for `mlp_out`."""

    @abstractmethod
    def mlp_gate_proj(self, layer_idx: int) -> nn.Module:
        """Hook its forward (output) for `mlp_gate`. A future architecture with a
        fused gate/up projection must still expose separate gate/up here — split
        the fused output by config, don't change this interface."""

    @abstractmethod
    def mlp_up_proj(self, layer_idx: int) -> nn.Module:
        """Hook its forward (output) for `mlp_up`."""

    @abstractmethod
    def mlp_down_proj(self, layer_idx: int) -> nn.Module:
        """Hook its forward-pre (input) for `mlp_act` — act_fn(gate) * up."""

    @abstractmethod
    def mlp_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """The model's own activation function (e.g. SiLU), read from a real
        submodule rather than looked up by name, so recompute matches exactly."""

    @property
    def mlp_activation_negative_min(self) -> float:
        """Infimum of `mlp_activation_fn` on (-inf, 0) — bounds the MLP-neuron
        'leak_neg'/'double_neg' regimes (Addendum A). Generic across
        architectures: looked up by `config.hidden_act`, not overridden per adapter."""
        hidden_act = self.config.hidden_act
        if hidden_act not in _ACTIVATION_NEGATIVE_MIN:
            raise NotImplementedError(f"negative-side minimum not known for activation {hidden_act!r}")
        return _ACTIVATION_NEGATIVE_MIN[hidden_act]

    # --- recompute primitives (architecture-specific) ---

    def reshape_to_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        """[batch, seq, num_heads * head_dim] -> [batch, num_heads, seq, head_dim].

        Standard reshape shared by most architectures; override if a model departs
        from it.
        """
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, num_heads, self.head_dim).transpose(1, 2)

    @abstractmethod
    def apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply this architecture's QK-norm (identity if the architecture has none)."""

    @abstractmethod
    def apply_rotary(
        self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE using the model's own rotary module(s) for this layer."""
