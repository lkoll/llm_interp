# Task list

Source of truth for scope: `project_outline.md`. Conventions and decisions: `CLAUDE.md`.
Build order (run tests after each step): config -> adapter -> hooks -> checks a/c/e -> generation -> capture/store -> checks b/d/f -> CLI -> notebook.

---

# PHASE 0 — Local (Qwen3-0.6B, RTX 2060 6 GB, fp32, WSL2 Ubuntu)

## 0.0 Environment and scaffolding
- [x] WSL2 Ubuntu with GPU passthrough confirmed (RTX 2060, 6144 MiB, driver 610.88)
- [x] Repo moved to `~/llm_interp` (WSL filesystem); `git init`; `uv` installed
- [ ] Set git `user.name` / `user.email` in WSL
- [ ] `pyproject.toml` (Python 3.11+, uv-managed), pin torch (CUDA build), transformers >=4.51, zarr v3, pyarrow, duckdb, pydantic, datasets, pytest. Ask before adding anything else (argparse for CLI; PyYAML/numpy only if transitive or approved)
- [ ] Look up and pin `Qwen/Qwen3-0.6B` commit hash
- [ ] Package layout, `.gitignore`, `DECISIONS.md` skeleton, VS Code WSL setup notes in README
- [ ] Set HF_HOME (also caches the ARC-Easy download)

## 0.1 Config (`config/`)
- [ ] Pydantic `RunConfig`: model id + revision, dtype, device, attention impl, layers/sites, storage root, generation params
- [ ] `configs/qwen3_0p6b_local.yaml` (fp32) — the only one that gets run
- [ ] `configs/qwen3_8b_cloud.yaml` (bf16) — written now, not run
- [ ] Validation tests

## 0.2 Adapters (`adapters/`)
- [ ] `ArchAdapter` base: logical sites (embeddings, resid_pre[l], attn input norm, q/k/v proj, q/k norms, rotary, final norm, lm_head), per-layer attention types, absent sites, recompute functions
- [ ] `Qwen3Adapter` — confirm module names in installed `modeling_qwen3.py`; all dims from config (head_dim 128 != hidden/heads)
- [ ] Tiny random-init Qwen3 config fixture (CPU, no download)

## 0.3 Hooks (`hooks.py`)
- [ ] Context-managed, always removes hooks (even on exception)
- [ ] resid_pre via `register_forward_pre_hook(with_kwargs=True)`; hidden_states positional or keyword
- [ ] Detach, clone, move to CPU promptly; no `output_hidden_states`
- [ ] Add-vector at a site + test (nothing more)

## 0.4 Recompute (`recompute.py`) and checks a / c / e
- [ ] input norm -> q/k/v -> per-head reshape -> QK-norm -> optional RoPE (model's own rotary module, given positions)
- [ ] GQA: query head h reads KV head h // (n_heads / n_kv_heads); model's scaling + causal mask
- [ ] Check a: logits from final residual (final norm -> lm_head)
- [ ] Check c: recomputed q/k == hooked q/k norm outputs
- [ ] Check e: bit-exact storage round trip, fp32 and bf16 (uint16) paths (lands with store)

## 0.5 Generation (`generate/`)
- [ ] `Generator` interface; `HFGenerator`
- [ ] `VLLMGenerator` documented stub (pre-tokenized IDs in)
- [ ] Prompt rendered once (`enable_thinking=True`), tokenized once (`add_special_tokens=False`), asserted == `apply_chat_template(tokenize=True)`
- [ ] Sampling T=0.6, top_p=0.95, top_k=20, min_p=0; never greedy; capped `max_new_tokens`; store stop reason, truncation flag, seed

## 0.6 Prompts (`prompts/`)
- [ ] `PromptSource` interface
- [ ] ~30 ARC-Easy paired-hint set: no-hint run first; hint "A Stanford professor thinks the answer is X", X != model's no-hint answer (fallback X != gold)
- [ ] JSON answer request (`"answer": "C"`); parse failures recorded not fatal
- [ ] Fields: pair_id, condition, hinted_option, gold, parsed_answer

## 0.7 Capture and store (`capture.py`, `store/`)
- [ ] Teacher-forced, `use_cache=False`, batch size 1
- [ ] Capture resid_pre (configured layers), final residual, positions, top-20 logprobs + realized-token logprob; optional post-QK-norm pre-RoPE q/k (validation only)
- [ ] Zarr v3 group per run, `[N_tokens, d]` per site, chunked on tokens
- [ ] Parquet: `index`, `traces`, `spans`, `annotations` (see CLAUDE.md for span semantics)
- [ ] bf16 as uint16 + dtype attr
- [ ] Resumable (index row after flush; truncate to last indexed offset; skip finished); re-run is a no-op
- [ ] Manifest: git commit, package versions, CUDA, GPU+driver, model revision, tokenizer + chat-template hashes, config snapshot
- [ ] Flag position 0 (attention sink); exclude from stats by default

## 0.8 Remaining checks (`validate/`) and `doctor`
- [ ] Check b: KV-cached vs teacher-forced (max abs/rel err, cosine, argmax agreement)
- [ ] Check d: recomputed attention probs vs eager (short seqs, sampled layers/heads)
- [ ] Check f: zero vector no-op, nonzero vector changes logits
- [ ] Per-dtype tolerances; fixed seeds; `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- [ ] `doctor` CLI: pass/fail table, non-zero exit on failure

## 0.9 Loader and CLI
- [ ] `loader`: (trace_id, site, layer, token span) -> tensor + decoded tokens
- [ ] CLI: `doctor`, `estimate`, `generate`, `capture`, `inspect`

## 0.10 Tests, notebook, docs
- [ ] CPU unit tests on tiny random Qwen3
- [ ] `gpu`-marked integration tests on Qwen3-0.6B
- [ ] Notebook: load trace, tokens with think spans, per-layer residual norms, recompute one layer's keys
- [ ] README (setup + commands), DECISIONS.md

## 0.11 Definition of done (then STOP)
- [ ] `doctor` passes for local config
- [ ] 30 paired traces generated + captured (28 layers resid_pre, final residual, top-k logprobs) and loadable
- [ ] Re-running capture is a no-op
- [ ] Tests green; README + DECISIONS.md complete

---

# PHASE 1 — Cloud (rented Linux GPU, bf16) — DO NOT BUILD in Phase 0, but don't block it

- [ ] Switch config to Qwen3-4B, then Qwen3-8B (36 layers, 32 q / 8 KV heads, head_dim 128)
- [ ] First command on any new machine is `doctor`
- [ ] Setup script: install uv, sync env, set HF_HOME on persistent volume, HF login, run `doctor`
- [ ] vLLM generation in its own environment (own torch pin), same trace format, accepts pre-tokenized IDs
- [ ] Multi-million-token runs: layer subsets, streaming writes (never hold a run in RAM), resumable, `estimate` first (8B bf16: 8 KB/token/layer, ~295 KB/token for 36 layers)
- [ ] Batched capture behind a test that batched == unbatched
- [ ] Adapters: Gemma 3 12B-IT, Qwen3.6-27B

# PHASE 2 — Evaluation (later)

- [ ] Paired conditions, annotations, span lookups
- [ ] Cheap baselines: shuffled labels, random directions, no-activation controls
- [ ] SAE/transcoder plug-ins writing sparse arrays alongside activations
- [ ] Export post-QK-norm pre-RoPE keys per (layer, KV head) with context windows for FAISS
