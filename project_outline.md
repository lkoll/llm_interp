# Brief: Activation capture pipeline (Phase 0)

## Context
This repo supports interpretability research on LLM reasoning: reading a model's internal state while it writes chain-of-thought. Planned experiments (NOT this phase) include decoding how the model holds propositions (believed vs. asserted), decoding attention queries as "what the model is searching for," and stress-testing activation decoders against known ground truth. All need one substrate: an exact, validated record of a model's internals over its own generated reasoning traces.

Phase 0 builds that substrate on Qwen3-0.6B on a local GPU. Design it so that moving to larger models on rented cloud GPUs is a config change, not a rewrite. Do not implement experiments.

## Environment
- Local GPU: RTX 2060 (Turing, compute capability 7.5; confirm VRAM with `nvidia-smi`).
  - No native bf16: run fp32 locally.
  - No FlashAttention (needs Ampere+): use `sdpa` for capture and `eager` only for attention validation.
  - No `torch.compile`.
- If the host is Windows, work inside WSL2 Ubuntu, with the repo on the WSL filesystem (not `/mnt/c`). The cloud target is Linux.
- Python 3.11+, `uv`. Pin torch, transformers (>=4.51 for Qwen3), zarr (v3), pyarrow, duckdb, pydantic, datasets, pytest. Ask before adding anything else.
- Model: `Qwen/Qwen3-0.6B`, pinned to a commit hash.
  - Read every dimension from the config; never hardcode.
  - Watch out: hidden size is 1024, but 16 query heads x head_dim 128 = 2048, so head_dim != hidden/heads.
  - 8 KV heads (GQA), tied embeddings, per-head QK-norm applied before RoPE, RoPE theta 1e6, no sliding window.
  - Confirm module names by reading the installed transformers modeling code.

## Design principles (these make the cloud move a config change)
1. Config-driven. One pydantic-validated YAML per run: model id + revision, dtype, device, attention implementation, layers/sites to capture, storage root, generation params. Write `configs/qwen3_0p6b_local.yaml` (fp32) and `configs/qwen3_8b_cloud.yaml` (bf16) now; only the first gets run.
2. Architecture adapters.
   - An interface that maps logical sites (embeddings, resid_pre[l], attention input norm, q/k/v projections, q/k norms, rotary embedding, final norm, lm_head) to modules, plus recompute functions.
   - Implement `Qwen3Adapter` only, but the interface must allow per-layer attention types and absent sites.
   - Future adapters: Gemma 3 (local/global layers with different RoPE bases, sliding window, extra norms around each sublayer, embedding scaling, custom attention scaling, multimodal wrapper) and Qwen3.6 (hybrid: Gated DeltaNet layers plus gated full-attention layers).
3. Generation and capture are separate stages that talk only through the trace store. Generation produces token IDs; capture replays them teacher-forced in one forward pass. This is valid because a causal LM's activations at position t depend only on tokens <= t.
4. Token IDs are the source of truth.
   - Never re-tokenize decoded text, and never re-render a conversation through the chat template for capture: Qwen3's template strips `<think>` content from earlier turns.
   - Render the prompt once (`enable_thinking=True`), tokenize once with `add_special_tokens=False`, and assert the result equals `apply_chat_template(..., tokenize=True)`. Store those IDs.
5. One hook manager handles both reading and writing. Capture now; also support adding a vector at a site, which is needed later for steering and activation-injection decoders. Implement add-vector with a test, nothing more.

## Build
1. `config/` and `adapters/` (base + Qwen3).
2. `hooks.py`
   - Context-managed, and always removed, even on exceptions.
   - Get resid_pre via `register_forward_pre_hook(..., with_kwargs=True)`; hidden_states may arrive positionally or as a keyword depending on version.
   - Detach, clone, and move captures to CPU promptly.
   - Don't rely on `output_hidden_states`: its last entry is post-final-norm, and the semantics shift between versions.
3. `generate/`
   - `Generator` interface; implement `HFGenerator` now.
   - `VLLMGenerator` is a documented stub only. It will run in its own environment (vLLM pins its own torch) and must accept pre-tokenized prompt IDs.
   - Thinking mode on. Sampling: temperature 0.6, top_p 0.95, top_k 20, min_p 0. Never greedy; it causes repetition loops.
   - Cap `max_new_tokens` (0.6B rambles). Store stop reason, truncation flag, and seed.
4. `prompts/`
   - A `PromptSource` interface plus a toy paired-hint set: about 30 ARC-Easy questions.
   - Run each with no hint first, then with "A Stanford professor thinks the answer is X," where X != the model's no-hint answer (fallback: X != gold).
   - Request the answer as JSON: `"answer": "C"`.
   - Fields: pair_id, condition, hinted_option, gold, parsed_answer. Parse failures are recorded, not fatal.
   - This exists to exercise the data model; the results don't matter.
5. `capture.py`
   - Teacher-forced forward pass, `use_cache=False`, batch size 1. Batched capture comes later, behind a test that batched == unbatched.
   - Capture resid_pre for configured layers, the final residual (input to the final norm), positions, and top-20 logprobs plus the realized next token's logprob. Full-vocab logits are too large to store.
   - Optionally capture post-QK-norm, pre-RoPE q/k, for validation runs only.
6. `store/`
   - One Zarr group per run. Each site is an array concatenated along the token axis, [N_tokens, d], chunked along tokens.
   - Parquet tables:
     - `index`: trace_id -> start/end offsets.
     - `traces`: metadata.
     - `spans`: positions of `<think>`, `</think>`, answer, and hint. Get special-token IDs via `convert_tokens_to_ids`; locate the hint via the tokenizer's offset mapping.
     - `annotations`: append-only, for LLM-judge labels later.
   - NumPy has no bfloat16: store bf16 as raw uint16 bits with a dtype attribute, and view back on load.
   - Resumable: append the index row only after data is flushed. On restart, truncate arrays to the last indexed offset and skip finished traces.
   - Per-run manifest: git commit, package versions, CUDA version, GPU + driver, model revision, tokenizer and chat-template hashes, config snapshot.
7. `recompute.py`
   - From stored resid_pre: input norm -> q/k/v -> per-head reshape -> QK-norm -> optional RoPE at given positions, using the model's own rotary module.
   - GQA mapping: query head h reads KV head h // (n_heads / n_kv_heads).
   - Attention scores use the model's scaling and causal mask.
8. `validate/` plus a `doctor` CLI that runs on any machine/config, prints a pass/fail table, and exits non-zero on failure. Checks:
   a. Logits rebuilt from the final residual (final norm -> lm_head) match the model's logits.
   b. Activations hooked during KV-cached generation match teacher-forced activations. Report max abs/rel error, cosine similarity, and argmax agreement.
   c. q/k recomputed from resid_pre match the hooked q/k norm outputs.
   d. Recomputed attention probabilities match the model's (eager attention, short sequences, sampled layers/heads).
   e. Storage round trip is bit-exact, for both the fp32 and bf16 paths.
   f. Adding a zero vector leaves logits unchanged; adding a nonzero vector changes them.
   - Tolerances are keyed by dtype: tight in fp32, looser in bf16, where cached decoding and full forward passes legitimately differ.
   - Use fixed seeds and deterministic algorithms where possible (`CUBLAS_WORKSPACE_CONFIG=:4096:8`).
9. `loader`: (trace_id, site, layer, token span) -> tensor, returned with decoded tokens.
10. CLI commands: `doctor`, `estimate` (projected bytes from config + token count; run before any capture), `generate`, `capture`, `inspect`.
11. Tests and notebook:
    - Unit tests on a tiny, randomly initialized Qwen3 config (CPU, no download).
    - Integration tests on Qwen3-0.6B, marked `gpu`.
    - One notebook: load a trace, show tokens with think spans, plot per-layer residual norms, recompute one layer's keys.

## Gotchas
- Position 0 is an attention sink with outsized activations. Flag it and exclude it from statistics by default.
- Positions are `arange(T)` for unpadded input. If padding is ever added, derive positions from the attention mask and store them.
- Keep captures on CPU. Eager attention is O(T^2), so use it for validation on short sequences only.
- fp16 is not a safe substitute for bf16 (overflow). Use fp32 locally.
- Never quantize the model. Record dtype per run, and never mix fp32 and bf16 captures in one analysis.

## Definition of done
- `doctor` passes for the local config.
- 30 paired toy traces are generated, captured (resid_pre for all 28 layers, final residual, top-k logprobs), and loadable.
- Re-running capture is a no-op.
- Tests are green. The README covers setup and commands, and DECISIONS.md logs design choices.

Stop there.

## Design for later (don't build these, but don't block them)
- Phase 1, rented Linux GPU, bf16:
  - Same repo. Switch the config to Qwen3-4B, then Qwen3-8B (36 layers, 32 query / 8 KV heads, head_dim 128).
  - The first command on any new machine is `doctor`.
  - A setup script: install uv, sync the environment, set HF_HOME on the persistent volume, log in to Hugging Face, run `doctor`.
  - vLLM generation in its own environment, writing the same trace format.
  - Runs of millions of tokens: layer subsets, streaming writes (never hold a run in RAM), resumable, estimate first. Qwen3-8B's residual stream in bf16 is 8 KB per token per layer, about 295 KB per token for all 36 layers.
  - Gemma 3 12B-IT and Qwen3.6-27B adapters come later.
- Phase 2, evaluation:
  - Paired conditions, annotations, and span lookups.
  - Cheap baselines: shuffled labels, random directions, no-activation controls.
  - Sparse-code encoders (SAEs/transcoders) as plug-ins that write sparse arrays alongside activations.
  - Export of post-QK-norm, pre-RoPE keys per (layer, KV head), with context windows, for FAISS indexing.

## Working style
- Build in this order: config -> adapter -> hooks -> checks a/c/e -> generation -> capture/store -> checks b/d/f -> CLI -> notebook. Run tests after each step.
- Ask before changing the storage format.
- Keep it simple: no distributed frameworks, no database beyond Parquet + DuckDB.

## Addendum A: MLP neuron capture (native units)

### Why
[1] argues that the model's own MLP neurons may be mostly interpretable without a trained dictionary (SAE/transcoder). Its audits of Qwen3-32B put about 90% of neuron "poles" at one main job. Caveats: the numbers shift with how thresholds are set, and [1] is an informal essay with its technical companion paper still pending.

Two consequences for this project:
- Native MLP neurons become a first-class unit, alongside the residual stream and (later) SAE/transcoder features. Phase 1 compares them directly.
- [1] reports a "double-negative" regime in gated MLP neurons. Tooling must keep sign regimes separable.

### Background: the Qwen3 MLP
- The MLP computes `down_proj( act_fn(gate_proj(x)) * up_proj(x) )`.
  - `act_fn` comes from `config.hidden_act`; for Qwen3 it is SiLU.
  - `x` is the post-attention-norm residual.
- A "neuron" is one coordinate of down_proj's input: `a = act_fn(g) * u`.
  - Width is `intermediate_size`: 3,072 for 0.6B, 12,288 for 8B (3x the hidden size).
- SiLU preserves sign (SiLU(z) = z * sigmoid(z)), so sign(a) = sign(g) * sign(u). That gives four regimes:

| g | u | a | regime |
|---|---|---|---|
| + | + | + | `pos` |
| + | - | - | `neg` |
| - | + | - (small) | `leak_neg` |
| - | - | + (small) | `double_neg` |

- For negative g, SiLU(g) >= about -0.278, so the last two regimes are bounded by about 0.278 * |u|.
- Given a and sign(g), the regime is fully determined. So store a plus one gate-sign bit per neuron rather than g and u.

### Changes
1. **Adapter sites (logical, per layer):**
   - `mlp_in` (input to gate/up; optional)
   - `mlp_gate` (gate_proj output, pre-activation)
   - `mlp_up` (up_proj output)
   - `mlp_act` (input to down_proj)
   - `mlp_out` (MLP output)
   - The adapter also exposes the activation function and its negative-side minimum: SiLU about -0.278; GELU-tanh (used by Gemma 3) about -0.17. Both preserve sign, so the regime logic carries over.
   - Future adapters may have a fused gate/up projection; the interface must allow splitting it by config.
2. **Hooks:**
   - `mlp_act`: forward-pre-hook on `down_proj`.
   - `mlp_gate` / `mlp_up`: forward hooks on `gate_proj` / `up_proj`.
   - `mlp_out`: forward hook on `mlp`.
   - Same rules as the other hooks: context-managed, detach/clone/move to CPU promptly.
3. **Storage modes (per run, in config):**
   - `dense`: `mlp_act[l]` as [N_tokens, d_mlp] in the run dtype, plus `mlp_gate_sign[l]` as packed bits [N_tokens, d_mlp/8] uint8 (np.packbits; 1 = gate negative).
   - `topk`: per (token, layer), keep k entries by |a| (default k = 64). Store an int16 index, the value in the run dtype, and a uint8 regime code.
   - `validation`: additionally store dense `mlp_gate`, `mlp_up`, and `mlp_out`. Used only for check g.
   - Layers are configurable. Defaults: local runs use dense on 4 layers (about 25/50/75% depth plus the last layer); cloud runs use topk on all layers plus dense on 4.
4. **Estimator** includes the MLP sites. Reference figures for sanity-checking it:
   - Qwen3-0.6B, fp32:
     - dense act: 12.3 KB per token-layer, plus 384 B of gate bits
     - topk (k = 64): about 448 B per token-layer
   - Qwen3-8B, bf16:
     - dense act: 24.6 KB per token-layer (about 3x the residual). All 36 layers come to about 885 KB/token, or about 885 GB per 1M tokens.
     - topk (k = 64): 320 B per token-layer, about 11.5 GB per 1M tokens across 36 layers.
5. **Validation check g (MLP consistency),** run on a validation run:
   - act_fn(captured gate) * captured up == captured `mlp_act` (bitwise in fp32; within tolerance in bf16).
   - down_proj(captured `mlp_act`) == captured `mlp_out`.
   - Regime derived from (sign(a), gate bit) == regime derived from full (g, u).
   - topk round trip: kept entries are reproduced exactly; report the fraction of L2 norm lost to dropped entries.
6. **Loader:**
   - `get_neuron(trace_id, layer, idx, span)` returns values plus regime codes.
   - `top_contexts(layer, idx, pole, n)` is an interface stub only; Phase 1 builds the index behind it. It must return generous preceding context (configurable, default >= 1,024 tokens), not short snippets.

### Definition of done (additions)
- Toy traces are captured with dense `mlp_act` on 4 layers, plus gate bits.
- The estimator is updated.
- Check g passes.

### Design for later
- **Phase 1 neuron index:** an index of each neuron's strongest activations, built separately for the positive and negative poles (per [1], both carry meaning) and split by regime.
- **Phase 1 comparison:** run each idea's readout on native neurons and on SAE/transcoder features side by side. Training mid-layer SAEs for Qwen3-8B becomes conditional on neurons underperforming.
- **Conventions for future neuron dashboards, from [1]:**
  - Show both poles.
  - Show long preceding context; short snippets made long-context neurons look polysemantic.
  - Expect "summary" neurons that fire on punctuation.

### References
[1] Nick Cammarata, "Neural Networks Are Already Clean," Goodfire. https://already-clean.goodfire.com/ (accessed 2026-10-15). Informal essay and working hypothesis. A technical companion paper on language-model neurons is forthcoming; check for it before Phase 1.