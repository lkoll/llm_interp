import numpy as np
import torch

from llm_interp.mlp import (
    REGIME_DOUBLE_NEG,
    REGIME_LEAK_NEG,
    REGIME_NEG,
    REGIME_POS,
    pack_gate_sign_bits,
    regime_from_signs,
    topk_by_abs,
    unpack_gate_sign_bits,
)


def test_regime_from_signs_all_four_cases():
    gate = torch.tensor([1.0, 1.0, -1.0, -1.0])
    act = torch.tensor([2.0, -2.0, -0.1, 0.1])  # pos, neg, leak_neg, double_neg
    regime = regime_from_signs(gate, act)
    assert regime.tolist() == [REGIME_POS, REGIME_NEG, REGIME_LEAK_NEG, REGIME_DOUBLE_NEG]
    assert regime.dtype == torch.uint8


def test_regime_from_signs_zero_gate_and_act_are_non_negative():
    gate = torch.tensor([0.0, 0.0])
    act = torch.tensor([0.0, -1.0])
    regime = regime_from_signs(gate, act)
    assert regime.tolist() == [REGIME_POS, REGIME_NEG]


def test_regime_from_signs_matches_silu_relationship():
    """sign(act) should equal sign(gate) * sign(up) for real SiLU-gated values,
    and regime_from_signs should classify consistently with that."""
    torch.manual_seed(0)
    gate = torch.randn(1000)
    up = torch.randn(1000)
    act = torch.nn.functional.silu(gate) * up
    regime = regime_from_signs(gate, act)

    for g, a, r in zip(gate.tolist(), act.tolist(), regime.tolist()):
        if g >= 0 and a >= 0:
            assert r == REGIME_POS
        elif g >= 0 and a < 0:
            assert r == REGIME_NEG
        elif g < 0 and a < 0:
            assert r == REGIME_LEAK_NEG
        else:
            assert r == REGIME_DOUBLE_NEG


def test_pack_unpack_gate_sign_bits_round_trip_divisible_by_8():
    gate = torch.tensor([[1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0]])
    packed = pack_gate_sign_bits(gate)
    assert packed.shape == (1, 1)
    unpacked = unpack_gate_sign_bits(packed, d_mlp=8)
    expected = (gate < 0).to(torch.uint8).numpy()
    np.testing.assert_array_equal(unpacked, expected)


def test_pack_unpack_gate_sign_bits_round_trip_not_divisible_by_8():
    torch.manual_seed(1)
    gate = torch.randn(3, 13)  # not a multiple of 8
    packed = pack_gate_sign_bits(gate)
    assert packed.shape == (3, 2)  # ceil(13/8) = 2
    unpacked = unpack_gate_sign_bits(packed, d_mlp=13)
    expected = (gate < 0).to(torch.uint8).numpy()
    np.testing.assert_array_equal(unpacked, expected)


def test_topk_by_abs_keeps_largest_magnitude_entries():
    act = torch.tensor([[1.0, -5.0, 2.0, -0.5, 4.0]])
    gate = torch.ones_like(act)
    indices, values, regime = topk_by_abs(act, gate, k=2)
    assert set(indices[0].tolist()) == {1, 4}  # |-5| and |4| are the two largest
    torch.testing.assert_close(values.abs().sort(descending=True).values, torch.tensor([[5.0, 4.0]]))
    assert indices.dtype == torch.int16
    assert regime.dtype == torch.uint8


def test_topk_by_abs_regime_matches_values_and_gate():
    act = torch.tensor([[3.0, -3.0]])
    gate = torch.tensor([[1.0, -1.0]])  # pos and leak_neg respectively
    indices, values, regime = topk_by_abs(act, gate, k=2)
    for idx, val, reg in zip(indices[0].tolist(), values[0].tolist(), regime[0].tolist()):
        expected = regime_from_signs(gate[0, idx], torch.tensor(val)).item()
        assert reg == expected


def test_topk_by_abs_clamps_k_to_d_mlp():
    act = torch.randn(1, 4)
    gate = torch.randn(1, 4)
    indices, values, regime = topk_by_abs(act, gate, k=100)
    assert indices.shape == (1, 4)
    assert values.shape == (1, 4)
    assert regime.shape == (1, 4)
