from llm_interp.validate import main, print_report
from llm_interp.validate.base import CheckResult


def test_print_report_shows_pass_fail_and_details(capsys):
    results = [
        CheckResult("check_a", True, {"max_abs_err": 0.0}),
        CheckResult("check_b", False, {"max_abs_err": 1.0}),
        CheckResult("check_c", False, {}, error="boom"),
    ]
    print_report(results)
    out = capsys.readouterr().out
    assert "check_a" in out and "PASS" in out
    assert "check_b" in out and "FAIL" in out
    assert "max_abs_err=1.0" in out
    assert "error: boom" in out
    assert "1/3 checks passed" in out


def test_main_exits_nonzero_on_any_failure(monkeypatch, capsys):
    def fake_run_doctor(config_path):
        return [
            CheckResult("check_a", True, {}),
            CheckResult("check_b", False, {}, error="simulated failure"),
        ]

    monkeypatch.setattr("llm_interp.validate.doctor.run_doctor", fake_run_doctor)
    exit_code = main(["--config", "irrelevant.yaml"])
    assert exit_code == 1
    assert "1/2 checks passed" in capsys.readouterr().out


def test_main_exits_zero_when_all_pass(monkeypatch):
    def fake_run_doctor(config_path):
        return [CheckResult("check_a", True, {}), CheckResult("check_b", True, {})]

    monkeypatch.setattr("llm_interp.validate.doctor.run_doctor", fake_run_doctor)
    assert main(["--config", "irrelevant.yaml"]) == 0
