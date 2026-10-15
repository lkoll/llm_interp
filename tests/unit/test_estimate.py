from llm_interp.estimate import format_bytes


def test_format_bytes_under_1kb():
    assert format_bytes(512) == "512.0 B"


def test_format_bytes_kb():
    assert format_bytes(2048) == "2.0 KB"


def test_format_bytes_mb():
    assert format_bytes(5 * 1024 * 1024) == "5.0 MB"


def test_format_bytes_gb():
    assert format_bytes(3 * 1024 * 1024 * 1024) == "3.0 GB"


def test_format_bytes_zero():
    assert format_bytes(0) == "0.0 B"
