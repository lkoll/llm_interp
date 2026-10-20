# Task list

Source of truth for scope: `project_outline.md`. Conventions and decisions: `CLAUDE.md`.
Build order (run tests after each step): config -> adapter -> hooks -> checks a/c/e -> generation -> capture/store -> checks b/d/f -> CLI -> notebook.

---

# PHASE 0 — Local (Qwen3-0.6B, RTX 2060 6 GB, fp32, WSL2 Ubuntu)

## 0.0 Environment and scaffolding
- [x] WSL2 Ubuntu with GPU passthrough confirmed (RTX 2060, 6144 MiB, driver 610.88)
- [x] Repo moved to `~/llm_interp` (WSL filesystem); `git init`; `uv` installed
- [x] Set git `user.name` / `user.email` in WSL
- [x] `pyproject.toml` (Python 3.11+, uv-managed), pin torch (CUDA build), transformers >=4.51, zarr v3, pyarrow, duckdb, pydantic, datasets, pytest. Ask before adding anything else (argparse for CLI; PyYAML/numpy only if transitive or approved)
- [x] Look up and pin `Qwen/Qwen3-0.6B` commit hash
- [x] Package layout, `.gitignore`, `DECISIONS.md` skeleton, VS Code WSL setup notes in README
- [x] Set HF_HOME (also caches the ARC-Easy download)

## 0.1 Config (`config/`)
- [x] Pydantic `RunConfig`: model id + revision, dtype, device, attention impl, layers/sites, storage root, generation params
- [x] `configs/qwen3_0p6b_local.yaml` (fp32) — the only one that gets run
- [x] `configs/qwen3_8b_cloud.yaml` (bf16) — written now, not run
- [x] Validation tests

## 0.2 Adapters (`adapters/`)
- [x] `ArchAdapter` base: logical sites (embeddings, resid_pre[l], attn input norm, q/k/v proj, q/k norms, rotary, final norm, lm_head), per-layer attention types, absent sites, recompute functions
- [x] `Qwen3Adapter` — confirm module names in installed `modeling_qwen3.py`; all dims from config (head_dim 128 != hidden/heads)
- [x] Tiny random-init Qwen3 config fixture (CPU, no download)

## 0.3 Hooks (`hooks.py`)
- [x] Context-managed, always removes hooks (even on exception)
- [x] resid_pre via `register_forward_pre_hook(with_kwargs=True)`; hidden_states positional or keyword
- [x] Detach, clone, move to CPU promptly; no `output_hidden_states`
- [x] Add-vector at a site + test (nothing more)

## 0.4 Recompute (`recompute.py`) and checks a / c / e
- [x] input norm -> q/k/v -> per-head reshape -> QK-norm -> optional RoPE (model's own rotary module, given positions)
- [x] GQA: query head h reads KV head h // (n_heads / n_kv_heads); model's scaling + causal mask
- [x] Check a: logits from final residual (final norm -> lm_head)
- [x] Check c: recomputed q/k == hooked q/k norm outputs
- [x] Check e: bit-exact storage round trip, fp32 and bf16 (uint16) paths (lands with store)

## 0.5 Generation (`generate/`)
- [x] `Generator` interface; `HFGenerator`
- [x] `VLLMGenerator` documented stub (pre-tokenized IDs in)
- [x] Prompt rendered once (`enable_thinking=True`), tokenized once (`add_special_tokens=False`), asserted == `apply_chat_template(tokenize=True)`
- [x] Sampling T=0.6, top_p=0.95, top_k=20, min_p=0; never greedy; capped `max_new_tokens`; store stop reason, truncation flag, seed

## 0.6 Prompts (`prompts/`)
- [x] `PromptSource` interface
- [x] ~30 ARC-Easy paired-hint set: no-hint run first; hint "A Stanford professor thinks the answer is X", X != model's no-hint answer (fallback X != gold)
- [x] JSON answer request (`"answer": "C"`); parse failures recorded not fatal
- [x] Fields: pair_id, condition, hinted_option, gold, parsed_answer

## 0.7 Capture and store (`capture.py`, `store/`)
- [x] Teacher-forced, `use_cache=False`, batch size 1
- [x] Capture resid_pre (configured layers), final residual, positions, top-20 logprobs + realized-token logprob; optional post-QK-norm pre-RoPE q/k (validation only)
- [x] Zarr v3 group per run, `[N_tokens, d]` per site, chunked on tokens
- [x] Parquet: `index`, `traces`, `spans`, `annotations` (see CLAUDE.md for span semantics)
- [x] bf16 as uint16 + dtype attr
- [x] Resumable (index row after flush; truncate to last indexed offset; skip finished); re-run is a no-op
- [x] Manifest: git commit, package versions, CUDA, GPU+driver, model revision, tokenizer + chat-template hashes, config snapshot
- [x] Flag position 0 (attention sink); exclude from stats by default (identifiable directly from stored `positions`; "exclude by default" behavior deferred to the loader, 0.9 — noted in DECISIONS.md)

## 0.8 Remaining checks (`validate/`) and `doctor`
- [x] Check b: KV-cached vs teacher-forced (max abs/rel err, cosine, argmax agreement)
- [x] Check d: recomputed attention probs vs eager (short seqs, sampled layers/heads)
- [x] Check f: zero vector no-op, nonzero vector changes logits
- [x] Per-dtype tolerances; fixed seeds; `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- [x] `doctor` CLI: pass/fail table, non-zero exit on failure

## 0.9 Loader and CLI
- [x] `loader`: (trace_id, site, layer, token span) -> tensor + decoded tokens
- [x] CLI: `doctor`, `estimate`, `generate`, `capture`, `inspect`

## 0.10 Tests, notebook, docs
- [x] CPU unit tests on tiny random Qwen3
- [x] `gpu`-marked integration tests on Qwen3-0.6B
- [x] Notebook: load trace, tokens with think spans, per-layer residual norms, recompute one layer's keys
- [x] README (setup + commands), DECISIONS.md

## 0.11 Definition of done (then STOP)
- [x] `doctor` passes for local config
- [x] 30 paired traces generated + captured (28 layers resid_pre, final residual, top-k logprobs) and loadable
- [x] Re-running capture is a no-op
- [x] Tests green; README + DECISIONS.md complete

---

# PHASE 0 ADDENDUM A — MLP neuron capture (native units)

Source: `project_outline.md` "Addendum A" (added 2026-10-15). Native MLP neurons become a first-class unit alongside the residual stream; Phase 1 compares them against SAE/transcoder features. Touches config, adapters, hooks, store, estimator, validate, and loader — not appended at the end, woven through the existing pieces. Same build order pattern as the rest of Phase 0 (config -> adapter -> hooks -> storage -> estimator -> check g -> loader); run tests after each step.

## 0.12 Config (MLP capture)
- [x] `RunConfig` capture section: `MlpCaptureConfig` with independent `dense_layers` / `topk_layers` (may overlap) and `top_k` (default 64)
- [x] `configs/qwen3_0p6b_local.yaml`: dense on layers 7, 14, 21, 27 (~25/50/75% depth of 28 + last)
- [x] `configs/qwen3_8b_cloud.yaml`: topk on all 36 layers + dense on 9, 18, 27, 35 — written, not run
- [x] Validation tests for the new fields

## 0.13 Adapters (MLP sites)
- [x] `ArchAdapter`: new logical sites `mlp_in` (optional), `mlp_gate`, `mlp_up`, `mlp_act` (down_proj input), `mlp_out`
- [x] Adapter exposes the activation function and its negative-side minimum (SiLU ≈ -0.278; note GELU-tanh ≈ -0.17 for a future Gemma 3 adapter — both sign-preserving)
- [x] `Qwen3Adapter`: confirm `gate_proj`/`up_proj`/`down_proj`/`mlp` module names in installed `modeling_qwen3.py`
- [x] Interface must allow a future fused gate/up projection to be split by config (Gemma 3 / Qwen3.6 forward-compat note; no implementation now)
- [x] Tests on the tiny random-init fixture

## 0.14 Hooks (MLP sites)
- [x] `mlp_act`: forward-pre-hook on `down_proj`
- [x] `mlp_gate` / `mlp_up`: forward hooks on `gate_proj` / `up_proj`
- [x] `mlp_out`: forward hook on `mlp`
- [x] Same discipline as existing hooks: context-managed, always removed, detach/clone/move to CPU promptly
- [x] Tests

## 0.15 Storage (dense / topk / validation modes)
- [x] `dense`: `mlp_act[l]` as `[N_tokens, d_mlp]` in run dtype; `mlp_gate_sign[l]` as packed bits `[N_tokens, d_mlp/8]` uint8 (`np.packbits`; 1 = gate negative) — **flags `numpy` as a new direct dependency** (previously transitive only; flag per CLAUDE.md's dependency policy when implementing)
- [x] `topk` (k=64 default): per (token, layer) — int16 index, value in run dtype, uint8 regime code, kept by `|a|`
- [x] `validation`: additionally dense `mlp_gate`, `mlp_up`, `mlp_out` (check g only)
- [x] Bit-exact round-trip tests for packed gate-sign bits and topk encoding (extends check e)

## 0.16 Estimator (MLP sites)
- [x] `estimate.py`: include dense act + gate bits, or topk, per configured MLP layer, in the per-site byte accounting
- [x] Sanity-check against the addendum's reference figures (0.6B fp32 dense: ~12.3 KB/token-layer + 384 B gate bits; topk k=64: ~448 B/token-layer; 8B bf16 dense: ~24.6 KB/token-layer, ~885 GB/1M tokens over 36 layers; topk k=64: ~320 B/token-layer, ~11.5 GB/1M tokens over 36 layers)

## 0.17 Check g (MLP consistency) and `doctor`
- [x] `act_fn(gate) * up == mlp_act` (bitwise fp32; tolerance in bf16)
- [x] `down_proj(mlp_act) == mlp_out`
- [x] Regime from `(sign(a), gate bit)` == regime from full `(g, u)`
- [x] topk round trip: kept entries reproduced exactly; report fraction of L2 norm lost to dropped entries
- [x] Wire check g into `doctor`'s pass/fail table (needs a `validation`-mode run to exercise `mlp_gate`/`mlp_up`/`mlp_out`)

## 0.18 Loader (neuron access)
- [x] `get_neuron(trace_id, layer, idx, span)` -> values + regime codes
- [x] `top_contexts(layer, idx, pole, n)` — interface stub only (Phase 1 builds the index); must default to >= 1,024 tokens of preceding context (configurable)

## 0.19 Definition of done (Addendum A)
- [x] Toy traces captured with dense `mlp_act` on 4 layers + gate bits
- [x] Estimator updated and sanity-checked against the addendum's reference figures
- [x] Check g passes

---

# PHASE 1 — Cloud (rented Linux GPU, bf16) — DO NOT BUILD in Phase 0, but don't block it

- [ ] Switch config to Qwen3-4B, then Qwen3-8B (36 layers, 32 q / 8 KV heads, head_dim 128)
- [ ] First command on any new machine is `doctor`
- [ ] vLLM generation in its own environment (own torch pin), same trace format, accepts pre-tokenized IDs
- [ ] Multi-million-token runs: layer subsets, streaming writes (never hold a run in RAM), resumable, `estimate` first (8B bf16: 8 KB/token/layer, ~295 KB/token for 36 layers)
- [ ] Batched capture behind a test that batched == unbatched
- [ ] Adapters: Gemma 3 12B-IT, Qwen3.6-27B
- [ ] Neuron index (Addendum A): per-neuron strongest activations, built separately for positive/negative poles, split by regime
- [ ] Neuron vs. SAE/transcoder comparison (Addendum A): run each idea's readout on native neurons and on SAE/transcoder features side by side; training mid-layer SAEs for Qwen3-8B is conditional on neurons underperforming
- [ ] Neuron dashboard conventions (Addendum A, from [1]): show both poles; show long preceding context (short snippets make long-context neurons look polysemantic); expect "summary" neurons firing on punctuation

## Addendum B: Cloud setup (RunPod)

Source: `project_outline.md` "Addendum B" (added 2026-10-15). RunPod Secure Cloud + one network volume. Supersedes the old generic "setup script" bullet above with a concrete breakdown. The manual steps are Luke's own one-time account/pod setup, not agent-buildable; everything else is a script or a `doctor` addition that ships with the repo.

**2026-10-20 update (see DECISIONS.md): runs are decoupled per model, not one long-lived shared volume.** Qwen3-8B and the later 27B-class work each get their own network volume, pulled down and deleted when that run is done. This drops the "one data center with both 48 GB and 80 GB tiers" constraint below — each volume only needs to live in a data center with good price/availability for *that run's* tier.

### Manual steps (Luke, once per cloud run — not agent work)
- [x] Private GitHub repo, pushed from WSL
- [x] SSH keys: one on Windows (VS Code Remote-SSH), one in WSL (rsync); both pasted into RunPod account settings
- [ ] Prepaid credit only, no auto-reload
- [ ] Network volume created per model/run, in whichever data center has the best price/availability for that run's GPU tier (48 GB for Qwen3-4B/8B; 80 GB only when a 27B-class run comes up) — no longer needs both tiers in one data center
- [ ] Pod created: official PyTorch template, Secure Cloud, volume at `/workspace`, `HF_TOKEN` as a pod env var
- [ ] VS Code Remote-SSH connected to `/workspace/llm_interp`; coding agent extension installed remotely

### Scripts (`scripts/`)
- [x] `cloud_setup.sh` (idempotent): check GPU via `nvidia-smi`; install uv if missing; export + persist `HF_HOME`/`UV_CACHE_DIR` to `~/.bashrc`; clone or pull the repo (via `GITHUB_TOKEN`, see DECISIONS.md); `uv sync` for the main env plus a separate, unpinned vLLM env; pre-download weights for the chosen config (`HF_TOKEN` from env, never written into the repo); run `doctor` with the cloud config and `--storage-root` derived from the config's `run_name` under `/workspace/runs/` — not yet run against a real pod
- [x] `run.sh <run_id> <command>`: run inside a named `tmux` session, logs to `/workspace/runs/<run_id>/logs/`; heartbeat file while running; `DONE` marker + exit code at the end; optional `--stop-pod-when-done` (`runpodctl stop pod "$RUNPOD_POD_ID"` — confirmed both the env var and the correct argument order, which is reversed from this bullet's original phrasing; skips with a warning if the CLI/env var isn't available) — tested locally end-to-end (success, failure, and the warning path) with real `tmux`
- [x] `pull_run.sh <run_id> <dest>`: rsync over full SSH pod -> local archive path; verified with an independent `sha256sum` pass (not just rsync's own transfer-time checksums); optional `--delete-remote-after-verify` — argument/usage paths tested locally; the real rsync/ssh path needs a live pod to verify (no local sshd to test against)
- [x] `backup_meta.sh`: copies the four Parquet tables, `manifest.json`, and `generated_traces.json` to a `meta/<run_name>/` folder (default), skipping `activations.zarr/` — tested against both a mock run and the real Phase 0 `runs/qwen3_0p6b_local` run on disk

### `doctor` additions (cloud-specific)
- [x] Report GPU, driver, CUDA version, bf16 support, free VRAM
- [x] Fail if `storage_root` or `HF_HOME` isn't on the network volume (data would vanish with the pod)
- [x] Benchmark write/read throughput on the store path
- [x] Compare the run's `estimate`d size against free space on the volume

All four gated on `RUNPOD_POD_ID` being set (skip-as-pass otherwise, same convention as check (g)'s MLP-config gate) — see DECISIONS.md. Verified both the skip path (CPU unit test) and the real logic (GPU integration test against this machine's RTX 2060, simulating pod env vars) — all local + GPU tests green (131 unit, 39 GPU).

### Storage policy
- [ ] Precious (back up every run): token IDs, metadata, spans, annotations, manifests, configs — small, via `backup_meta.sh`
- [ ] Regenerable (treat as cache): activations — re-capturable from stored token IDs at the same model revision + dtype (manifest records hardware, since values may differ slightly across GPU types)
- [ ] Zarr chunk size ~4,096 tokens (16 MB for 0.6B fp32 residual stream; 32 MB for 8B bf16) for sequential reads on archive drives — note this is a *larger* chunk than Phase 0's current chunking; reconcile when implementing
- [ ] Finished cloud runs archived to local 2 TB drive via `pull_run.sh`, then deleted from the volume — per the 2026-10-20 decoupling decision, once a model's run is fully pulled and verified with no further runs planned on that volume, delete the whole volume (not just prune files from it) to stop the per-GB/month charge

### Cost guardrails
- [ ] Terminate pods when idle (volume persists)
- [ ] Default to 48 GB cards; 80 GB only for 27B-class runs
- [ ] Monthly check of volume usage; prune already-pulled runs, or delete the volume outright once a model's run is fully archived locally

# PHASE 2 — Evaluation (later)

- [ ] Paired conditions, annotations, span lookups
- [ ] Cheap baselines: shuffled labels, random directions, no-activation controls
- [ ] SAE/transcoder plug-ins writing sparse arrays alongside activations
- [ ] Export post-QK-norm pre-RoPE keys per (layer, KV head) with context windows for FAISS
