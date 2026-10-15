import tempfile
from pathlib import Path

import torch

from ..hooks import HookManager
from ..recompute import recompute_attention, recompute_logits, recompute_qk, recompute_qk_pre_rope, recompute_v
from .base import CACHE_VS_TEACHER_FORCED_ATOL, MIN_COSINE_SIMILARITY, CheckResult, ValidationContext


def _sample_layers(num_layers: int, k: int = 3) -> list[int]:
    """First, middle, last — deterministic, cheap enough for O(T^2) eager attention."""
    if num_layers <= k:
        return list(range(num_layers))
    return sorted({0, num_layers // 2, num_layers - 1})


def check_a_logits_from_final_residual(ctx: ValidationContext, input_ids: torch.Tensor) -> CheckResult:
    """Logits rebuilt from the final residual (final norm -> lm_head) match the
    model's own logits."""
    try:
        ids = input_ids.to(ctx.device)
        mgr = HookManager(ctx.adapter).capture_final_residual()
        with mgr, torch.no_grad():
            out = ctx.model(input_ids=ids)

        recomputed = recompute_logits(ctx.adapter, mgr.captures["final_residual"])
        diff = (recomputed - out.logits).abs()
        max_abs = diff.max().item()
        max_rel = (diff / (out.logits.abs() + 1e-8)).max().item()
        tol = ctx.tolerance
        passed = torch.allclose(recomputed, out.logits, atol=tol.atol, rtol=tol.rtol)
        return CheckResult(
            "check_a_logits_from_final_residual", passed, {"max_abs_err": max_abs, "max_rel_err": max_rel}
        )
    except Exception as e:
        return CheckResult("check_a_logits_from_final_residual", False, {}, error=str(e))


def check_c_recomputed_qk(ctx: ValidationContext, input_ids: torch.Tensor) -> CheckResult:
    """q/k recomputed from resid_pre (input norm -> proj -> reshape -> QK-norm)
    match the hooked q_norm/k_norm outputs, across every configured layer."""
    try:
        ids = input_ids.to(ctx.device)
        layers = list(range(ctx.adapter.num_layers))
        ground_truth: dict[int, list[torch.Tensor | None]] = {layer_idx: [None, None] for layer_idx in layers}

        def make_hook(layer_idx: int, slot: int):
            def hook(module, args, output):
                ground_truth[layer_idx][slot] = output.detach().clone()

            return hook

        handles = []
        for layer_idx in layers:
            handles.append(ctx.adapter.q_norm(layer_idx).register_forward_hook(make_hook(layer_idx, 0)))
            handles.append(ctx.adapter.k_norm(layer_idx).register_forward_hook(make_hook(layer_idx, 1)))

        mgr = HookManager(ctx.adapter).capture_resid_pre(layers)
        try:
            with mgr, torch.no_grad():
                ctx.model(input_ids=ids)
        finally:
            for h in handles:
                h.remove()

        tol = ctx.tolerance
        max_abs = 0.0
        max_rel = 0.0
        all_passed = True
        for layer_idx in layers:
            resid_pre = mgr.captures[layer_idx]
            q, k = recompute_qk_pre_rope(ctx.adapter, resid_pre, layer_idx)
            expected_q, expected_k = ground_truth[layer_idx]
            for actual, expected in ((q, expected_q.transpose(1, 2)), (k, expected_k.transpose(1, 2))):
                diff = (actual - expected).abs()
                max_abs = max(max_abs, diff.max().item())
                max_rel = max(max_rel, (diff / (expected.abs() + 1e-8)).max().item())
                if not torch.allclose(actual, expected, atol=tol.atol, rtol=tol.rtol):
                    all_passed = False
        return CheckResult(
            "check_c_recomputed_qk",
            all_passed,
            {"max_abs_err": max_abs, "max_rel_err": max_rel, "num_layers_checked": len(layers)},
        )
    except Exception as e:
        return CheckResult("check_c_recomputed_qk", False, {}, error=str(e))


def _capture_sites(ctx: ValidationContext, mgr: HookManager, layers: list[int]) -> dict:
    sites = {layer_idx: mgr.captures[layer_idx][0] for layer_idx in layers}
    sites["final_residual"] = mgr.captures["final_residual"][0]
    return sites


def _teacher_forced_capture(ctx: ValidationContext, full_ids: torch.Tensor, layers: list[int]) -> dict:
    ids = full_ids.unsqueeze(0).to(ctx.device)
    mgr = HookManager(ctx.adapter).capture_resid_pre(layers).capture_final_residual()
    with mgr, torch.no_grad():
        ctx.model(input_ids=ids, use_cache=False)
    return _capture_sites(ctx, mgr, layers)


def _kv_cached_capture(ctx: ValidationContext, full_ids: torch.Tensor, layers: list[int], n_decode_steps: int = 5) -> dict:
    ids = full_ids.to(ctx.device)
    n = ids.shape[0]
    n_decode_steps = min(n_decode_steps, n - 1)
    prefill_len = n - n_decode_steps

    pieces: dict = {layer_idx: [] for layer_idx in layers}
    pieces["final_residual"] = []

    prefill_ids = ids[:prefill_len].unsqueeze(0)
    mgr = HookManager(ctx.adapter).capture_resid_pre(layers).capture_final_residual()
    with mgr, torch.no_grad():
        out = ctx.model(input_ids=prefill_ids, use_cache=True)
    for layer_idx in layers:
        pieces[layer_idx].append(mgr.captures[layer_idx][0])
    pieces["final_residual"].append(mgr.captures["final_residual"][0])
    past_key_values = out.past_key_values

    for i in range(prefill_len, n):
        token = ids[i : i + 1].unsqueeze(0)
        step_mgr = HookManager(ctx.adapter).capture_resid_pre(layers).capture_final_residual()
        with step_mgr, torch.no_grad():
            out = ctx.model(input_ids=token, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        for layer_idx in layers:
            pieces[layer_idx].append(step_mgr.captures[layer_idx][0])
        pieces["final_residual"].append(step_mgr.captures["final_residual"][0])

    return {key: torch.cat(parts, dim=0) for key, parts in pieces.items()}


def check_b_kv_cache_vs_teacher_forced(ctx: ValidationContext, full_ids: torch.Tensor) -> CheckResult:
    """Activations hooked during KV-cached generation match teacher-forced
    activations: max abs/rel error, cosine similarity, and argmax agreement."""
    try:
        layers = ctx.config.capture.layers or list(range(ctx.adapter.num_layers))
        tf = _teacher_forced_capture(ctx, full_ids, layers)
        kv = _kv_cached_capture(ctx, full_ids, layers)

        # Absolute-error-only gate (see base.py: relative error is misleading near
        # zero, and this check compares two different computational paths, not a
        # recomputation of the same one — some drift is expected even in fp32).
        abs_tol = CACHE_VS_TEACHER_FORCED_ATOL[ctx.config.model.dtype]
        per_site: dict[str, dict] = {}
        all_passed = True
        for site in [*layers, "final_residual"]:
            a, b = tf[site], kv[site]
            diff = (a - b).abs()
            max_abs = diff.max().item()
            max_rel = (diff / (a.abs() + 1e-8)).max().item()
            cos_sim = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
            site_passed = max_abs <= abs_tol and cos_sim >= MIN_COSINE_SIMILARITY
            all_passed = all_passed and site_passed
            per_site[str(site)] = {"max_abs_err": max_abs, "max_rel_err": max_rel, "cosine_similarity": cos_sim}

        tf_logits = recompute_logits(ctx.adapter, tf["final_residual"])
        kv_logits = recompute_logits(ctx.adapter, kv["final_residual"])
        argmax_agreement = (tf_logits.argmax(-1) == kv_logits.argmax(-1)).float().mean().item()
        all_passed = all_passed and argmax_agreement == 1.0

        return CheckResult(
            "check_b_kv_cache_vs_teacher_forced",
            all_passed,
            {"per_site": per_site, "argmax_agreement": argmax_agreement},
        )
    except Exception as e:
        return CheckResult("check_b_kv_cache_vs_teacher_forced", False, {}, error=str(e))


def check_d_attention_probs_vs_eager(
    ctx: ValidationContext, input_ids: torch.Tensor, sampled_layers: list[int] | None = None
) -> CheckResult:
    """Recomputed attention probabilities match the model's own (eager attention,
    short sequences, sampled layers). Ground truth comes from `output_attentions`,
    which this model supports returning directly, rather than hooking a raw
    self_attn submodule that isn't one of the adapter's exposed logical sites."""
    try:
        sampled_layers = sampled_layers or _sample_layers(ctx.adapter.num_layers)
        original_impl = ctx.model.config._attn_implementation
        ctx.model.config._attn_implementation = "eager"
        try:
            ids = input_ids.to(ctx.device)
            mgr = HookManager(ctx.adapter).capture_resid_pre(sampled_layers)
            with mgr, torch.no_grad():
                out = ctx.model(input_ids=ids, output_attentions=True, use_cache=False)
        finally:
            ctx.model.config._attn_implementation = original_impl

        seq_len = ids.shape[-1]
        position_ids = torch.arange(seq_len, device=ids.device).unsqueeze(0)

        tol = ctx.tolerance
        max_abs = 0.0
        max_rel = 0.0
        all_passed = True
        for layer_idx in sampled_layers:
            resid_pre = mgr.captures[layer_idx]
            q, k = recompute_qk(ctx.adapter, resid_pre, layer_idx, position_ids)
            v = recompute_v(ctx.adapter, resid_pre, layer_idx)
            _, attn_weights = recompute_attention(ctx.adapter, q, k, v, layer_idx)
            expected = out.attentions[layer_idx]
            diff = (attn_weights - expected).abs()
            max_abs = max(max_abs, diff.max().item())
            max_rel = max(max_rel, (diff / (expected.abs() + 1e-8)).max().item())
            if not torch.allclose(attn_weights, expected, atol=tol.atol, rtol=tol.rtol):
                all_passed = False

        return CheckResult(
            "check_d_attention_probs_vs_eager",
            all_passed,
            {"max_abs_err": max_abs, "max_rel_err": max_rel, "sampled_layers": sampled_layers},
        )
    except Exception as e:
        return CheckResult("check_d_attention_probs_vs_eager", False, {}, error=str(e))


def check_e_storage_round_trip() -> CheckResult:
    """Storage round trip is bit-exact, for both the fp32 and bf16 paths."""
    try:
        from ..store import ZarrStore

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = ZarrStore(Path(tmp_dir) / "doctor_check_e.zarr")

            fp32_data = torch.randn(4, 8, dtype=torch.float32)
            s, e = store.append("fp32_site", fp32_data)
            fp32_ok = torch.equal(store.read("fp32_site", s, e), fp32_data)

            bf16_data = torch.randn(4, 8, dtype=torch.float32).to(torch.bfloat16)
            s2, e2 = store.append("bf16_site", bf16_data)
            read_back = store.read("bf16_site", s2, e2)
            bf16_ok = read_back.dtype == torch.bfloat16 and torch.equal(read_back, bf16_data)

        return CheckResult(
            "check_e_storage_round_trip", fp32_ok and bf16_ok, {"fp32_bit_exact": fp32_ok, "bf16_bit_exact": bf16_ok}
        )
    except Exception as e:
        return CheckResult("check_e_storage_round_trip", False, {}, error=str(e))


def check_f_add_vector(ctx: ValidationContext, input_ids: torch.Tensor, layer_idx: int = 0) -> CheckResult:
    """Adding a zero vector leaves logits unchanged; adding a nonzero vector
    changes them."""
    try:
        ids = input_ids.to(ctx.device)
        with torch.no_grad():
            baseline = ctx.model(input_ids=ids).logits

        zero_vector = torch.zeros(ctx.adapter.hidden_size, device=ctx.device)
        with HookManager(ctx.adapter).add_vector(layer_idx, zero_vector):
            with torch.no_grad():
                zero_logits = ctx.model(input_ids=ids).logits
        zero_is_noop = torch.equal(zero_logits, baseline)

        nonzero_vector = torch.ones(ctx.adapter.hidden_size, device=ctx.device) * 5.0
        with HookManager(ctx.adapter).add_vector(layer_idx, nonzero_vector):
            with torch.no_grad():
                nonzero_logits = ctx.model(input_ids=ids).logits
        tol = ctx.tolerance
        nonzero_changes = not torch.allclose(nonzero_logits, baseline, atol=tol.atol, rtol=tol.rtol)

        passed = zero_is_noop and nonzero_changes
        return CheckResult(
            "check_f_add_vector",
            passed,
            {"zero_is_noop": zero_is_noop, "nonzero_changes_logits": nonzero_changes, "layer_idx": layer_idx},
        )
    except Exception as e:
        return CheckResult("check_f_add_vector", False, {}, error=str(e))
