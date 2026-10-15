from collections.abc import Callable

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from .base import ArchAdapter


class Qwen3Adapter(ArchAdapter):
    @property
    def num_layers(self) -> int:
        return self.config.num_hidden_layers

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    @property
    def num_attention_heads(self) -> int:
        return self.config.num_attention_heads

    @property
    def num_key_value_heads(self) -> int:
        return self.config.num_key_value_heads

    @property
    def head_dim(self) -> int:
        return self.config.head_dim

    def embeddings(self) -> nn.Module:
        return self.model.model.embed_tokens

    def layer(self, layer_idx: int) -> nn.Module:
        self._check_layer_idx(layer_idx)
        return self.model.model.layers[layer_idx]

    def attn_input_norm(self, layer_idx: int) -> nn.Module:
        self._check_layer_idx(layer_idx)
        return self.model.model.layers[layer_idx].input_layernorm

    def q_proj(self, layer_idx: int) -> nn.Module:
        self._check_layer_idx(layer_idx)
        return self.model.model.layers[layer_idx].self_attn.q_proj

    def k_proj(self, layer_idx: int) -> nn.Module:
        self._check_layer_idx(layer_idx)
        return self.model.model.layers[layer_idx].self_attn.k_proj

    def v_proj(self, layer_idx: int) -> nn.Module:
        self._check_layer_idx(layer_idx)
        return self.model.model.layers[layer_idx].self_attn.v_proj

    def q_norm(self, layer_idx: int) -> nn.Module:
        self._check_layer_idx(layer_idx)
        return self.model.model.layers[layer_idx].self_attn.q_norm

    def k_norm(self, layer_idx: int) -> nn.Module:
        self._check_layer_idx(layer_idx)
        return self.model.model.layers[layer_idx].self_attn.k_norm

    def rotary_emb(self, layer_idx: int) -> nn.Module:
        self._check_layer_idx(layer_idx)
        return self.model.model.rotary_emb  # one shared rotary module for all layers

    def final_norm(self) -> nn.Module:
        return self.model.model.norm

    def lm_head(self) -> nn.Module:
        return self.model.lm_head

    def attention_type(self, layer_idx: int) -> str:
        self._check_layer_idx(layer_idx)
        return self.config.layer_types[layer_idx]

    def attention_scaling(self, layer_idx: int) -> float:
        self._check_layer_idx(layer_idx)
        return self.model.model.layers[layer_idx].self_attn.scaling

    def mlp(self, layer_idx: int) -> nn.Module:
        self._check_layer_idx(layer_idx)
        return self.model.model.layers[layer_idx].mlp

    def mlp_gate_proj(self, layer_idx: int) -> nn.Module:
        self._check_layer_idx(layer_idx)
        return self.model.model.layers[layer_idx].mlp.gate_proj

    def mlp_up_proj(self, layer_idx: int) -> nn.Module:
        self._check_layer_idx(layer_idx)
        return self.model.model.layers[layer_idx].mlp.up_proj

    def mlp_down_proj(self, layer_idx: int) -> nn.Module:
        self._check_layer_idx(layer_idx)
        return self.model.model.layers[layer_idx].mlp.down_proj

    def mlp_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return self.model.model.layers[0].mlp.act_fn

    def apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q_norm(layer_idx)(q), self.k_norm(layer_idx)(k)

    def apply_rotary(
        self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.rotary_emb(layer_idx)(q, position_ids)
        return apply_rotary_pos_emb(q, k, cos, sin)
