import numpy as np
import torch

# Four regimes from Addendum A's table, determined by (sign(gate), sign(act)):
# SiLU preserves sign, so sign(act) = sign(gate) * sign(up).
REGIME_POS = 0  # gate >= 0, act >= 0
REGIME_NEG = 1  # gate >= 0, act <  0
REGIME_LEAK_NEG = 2  # gate <  0, act <  0  (small, bounded by |act_fn's negative-side minimum| * |up|)
REGIME_DOUBLE_NEG = 3  # gate <  0, act >= 0  (small, same bound)

REGIME_NAMES = {
    REGIME_POS: "pos",
    REGIME_NEG: "neg",
    REGIME_LEAK_NEG: "leak_neg",
    REGIME_DOUBLE_NEG: "double_neg",
}


def regime_from_signs(gate: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
    """Regime code (uint8), same shape as `gate`/`act`. Needs only the raw gate
    pre-activation and the resulting neuron activation `act = act_fn(gate) * up`
    — not `up` itself — since the regime is fully determined by their two signs."""
    gate_negative = gate < 0
    act_negative = act < 0
    regime = torch.zeros_like(gate, dtype=torch.uint8)
    regime = torch.where(~gate_negative & act_negative, torch.full_like(regime, REGIME_NEG), regime)
    regime = torch.where(gate_negative & act_negative, torch.full_like(regime, REGIME_LEAK_NEG), regime)
    regime = torch.where(gate_negative & ~act_negative, torch.full_like(regime, REGIME_DOUBLE_NEG), regime)
    return regime


def pack_gate_sign_bits(gate: torch.Tensor) -> np.ndarray:
    """[..., d_mlp] gate pre-activations -> packed bits [..., ceil(d_mlp/8)]
    uint8 (`np.packbits`, last axis; 1 = gate negative). Invert with
    `unpack_gate_sign_bits`."""
    gate_negative = (gate < 0).to(torch.uint8).cpu().numpy()
    return np.packbits(gate_negative, axis=-1)


def unpack_gate_sign_bits(packed: np.ndarray, d_mlp: int) -> np.ndarray:
    """Inverse of `pack_gate_sign_bits`; `d_mlp` trims the padding bits
    `np.packbits` adds when `d_mlp` isn't a multiple of 8."""
    return np.unpackbits(packed, axis=-1, count=d_mlp)


def topk_by_abs(act: torch.Tensor, gate: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`act`, `gate`: [..., d_mlp]. Keeps the top `k` entries by `|act|`.
    Returns (indices [..., k] int16, values [..., k] same dtype as `act`,
    regime [..., k] uint8)."""
    k = min(k, act.shape[-1])
    _, indices = act.abs().topk(k, dim=-1)
    values = act.gather(-1, indices)
    gate_at_indices = gate.gather(-1, indices)
    regime = regime_from_signs(gate_at_indices, values)
    return indices.to(torch.int16), values, regime
