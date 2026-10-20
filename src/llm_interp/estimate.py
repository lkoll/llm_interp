from dataclasses import dataclass, field

from transformers import AutoConfig

from .config import RunConfig

# top_logprob_values/realized_logprob are always cast to float32 for log_softmax
# stability before storage, regardless of the model's own dtype (capture.py).
_LOGPROB_DTYPE_BYTES = 4
_INT64_BYTES = 8
_DTYPE_BYTES = {"float32": 4, "bfloat16": 2}


@dataclass(frozen=True)
class EstimateResult:
    per_site_bytes: dict[str, int] = field(default_factory=dict)
    bytes_per_token: int = 0
    num_tokens: int = 0
    total_bytes: int = 0


def estimate_bytes(config: RunConfig, num_tokens: int) -> EstimateResult:
    """Projected storage bytes from the config + an expected token count — run
    before any capture. Only fetches the model's (small) config.json via
    `AutoConfig`, never the model weights, since this needs to run cheaply
    ahead of a potentially very large capture."""
    model_config = AutoConfig.from_pretrained(config.model.id, revision=config.model.revision)
    hidden_size = model_config.hidden_size
    dtype_bytes = _DTYPE_BYTES[config.model.dtype]

    per_site: dict[str, int] = {}
    for layer in config.capture.layers:
        per_site[f"resid_pre/{layer}"] = hidden_size * dtype_bytes
    if config.capture.capture_final_residual:
        per_site["final_residual"] = hidden_size * dtype_bytes
    per_site["positions"] = _INT64_BYTES
    per_site["token_ids"] = _INT64_BYTES
    top_k = config.capture.top_k_logprobs
    per_site["top_logprob_values"] = top_k * _LOGPROB_DTYPE_BYTES
    per_site["top_logprob_indices"] = top_k * _INT64_BYTES
    per_site["realized_logprob"] = _LOGPROB_DTYPE_BYTES

    if config.capture.capture_qk_pre_rope:
        head_dim = model_config.head_dim
        num_heads = model_config.num_attention_heads
        num_kv_heads = model_config.num_key_value_heads
        for layer in config.capture.layers:
            per_site[f"qk_pre_rope/{layer}_q"] = num_heads * head_dim * dtype_bytes
            per_site[f"qk_pre_rope/{layer}_k"] = num_kv_heads * head_dim * dtype_bytes

    mlp_cfg = config.capture.mlp
    if mlp_cfg is not None:
        d_mlp = model_config.intermediate_size
        gate_sign_bytes = -(-d_mlp // 8)  # ceil(d_mlp / 8), np.packbits packing
        for layer in mlp_cfg.dense_layers:
            per_site[f"mlp_act/{layer}"] = d_mlp * dtype_bytes
            per_site[f"mlp_gate_sign/{layer}"] = gate_sign_bytes
        for layer in mlp_cfg.topk_layers:
            per_site[f"mlp_topk_index/{layer}"] = mlp_cfg.top_k * 2  # int16
            per_site[f"mlp_topk_value/{layer}"] = mlp_cfg.top_k * dtype_bytes
            per_site[f"mlp_topk_regime/{layer}"] = mlp_cfg.top_k * 1  # uint8
        if mlp_cfg.capture_validation_extras:
            for layer in sorted(set(mlp_cfg.dense_layers) | set(mlp_cfg.topk_layers)):
                per_site[f"mlp_gate/{layer}"] = d_mlp * dtype_bytes
                per_site[f"mlp_up/{layer}"] = d_mlp * dtype_bytes
                per_site[f"mlp_out/{layer}"] = hidden_size * dtype_bytes

    bytes_per_token = sum(per_site.values())
    return EstimateResult(
        per_site_bytes=per_site,
        bytes_per_token=bytes_per_token,
        num_tokens=num_tokens,
        total_bytes=bytes_per_token * num_tokens,
    )


def default_tokens_per_trace(config: RunConfig) -> int:
    """Nominal per-trace token count when the caller doesn't have a real one yet
    (e.g. before any generation has run): the generation cap plus a rough prompt
    length. Shared by `estimate`'s CLI and doctor's estimate-vs-free-space check
    so the two don't drift apart on what "a trace" is assumed to cost."""
    return config.generation.max_new_tokens + 100


def format_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def print_estimate(result: EstimateResult) -> None:
    grouped: dict[str, int] = {}
    for site, n_bytes in result.per_site_bytes.items():
        prefix = site.split("/")[0]
        grouped[prefix] = grouped.get(prefix, 0) + n_bytes

    print("Projected storage (per token, grouped by site family):")
    for prefix, n_bytes in sorted(grouped.items(), key=lambda kv: -kv[1]):
        print(f"  {prefix:<20} {format_bytes(n_bytes)}/token")
    print(f"\n  bytes/token (total):  {result.bytes_per_token} ({format_bytes(result.bytes_per_token)})")
    print(f"  num_tokens:           {result.num_tokens}")
    print(f"  projected total:      {format_bytes(result.total_bytes)}")
