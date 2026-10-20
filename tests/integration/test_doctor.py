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
        "check_cloud_gpu_report",
        "check_cloud_network_volume",
        "check_cloud_storage_throughput",
        "check_cloud_estimate_vs_free_space",
    }
    failures = [r for r in results if not r.passed]
    assert not failures, failures
    # Not running on a RunPod pod here, so the four cloud checks report skipped.
    cloud_results = [r for r in results if r.name.startswith("check_cloud_")]
    assert len(cloud_results) == 4
    assert all("skipped" in r.details for r in cloud_results)


def test_main_exits_zero_on_success(capsys):
    exit_code = main(["--config", "configs/qwen3_0p6b_local.yaml"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "11/11 checks passed" in captured.out
