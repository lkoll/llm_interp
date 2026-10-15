import pytest

from llm_interp.store.manifest import build_manifest

pytestmark = pytest.mark.gpu


def test_build_manifest_with_real_tokenizer(qwen3_model_and_tokenizer, qwen3_run_config):
    _, tokenizer = qwen3_model_and_tokenizer
    manifest = build_manifest(qwen3_run_config, tokenizer)

    assert manifest["model_id"] == "Qwen/Qwen3-0.6B"
    assert manifest["model_revision"] == qwen3_run_config.model.revision
    assert len(manifest["tokenizer_hash"]) == 64  # sha256 hex digest
    assert len(manifest["chat_template_hash"]) == 64
    assert manifest["package_versions"]["transformers"] is not None
    # git_commit may be None outside a git repo, but this project is one
    assert manifest["git_commit"] is not None
    assert manifest["gpu"]["name"] is not None  # nvidia-smi is available on this machine
