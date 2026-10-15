import pytest

from llm_interp.validate.doctor import main, run_doctor

pytestmark = pytest.mark.gpu


def test_run_doctor_all_checks_pass_on_local_config():
    results = run_doctor("configs/qwen3_0p6b_local.yaml")
    names = {r.name for r in results}
    assert names == {
        "check_a_logits_from_final_residual",
        "check_b_kv_cache_vs_teacher_forced",
        "check_c_recomputed_qk",
        "check_d_attention_probs_vs_eager",
        "check_e_storage_round_trip",
        "check_f_add_vector",
        "check_g_mlp_consistency",
    }
    failures = [r for r in results if not r.passed]
    assert not failures, failures


def test_main_exits_zero_on_success(capsys):
    exit_code = main(["--config", "configs/qwen3_0p6b_local.yaml"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "7/7 checks passed" in captured.out
