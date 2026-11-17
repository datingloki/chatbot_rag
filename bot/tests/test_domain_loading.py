"""Загрузка модулей по пути и проверка dev — без настоящей модели."""
import json

import pytest

from polybot.clients.domain import check_dev, load_module


def test_load_module_refuses_name_clash(tmp_path):
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_text("X = 1\n")
    b.write_text("X = 2\n")
    assert load_module("polybot_test_clash", a).X == 1
    assert load_module("polybot_test_clash", a).X == 1          # повторно — тот же модуль
    with pytest.raises(ImportError, match="уже загружен"):
        load_module("polybot_test_clash", b)


def _dev(tmp_path, rows):
    d = tmp_path / "dev set"
    d.mkdir()
    p = d / "dev.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return "dev set/dev.jsonl"


ROWS = [{"primary_class": "K1", "tags": []}, {"primary_class": "K2", "tags": []},
        {"primary_class": "K3", "tags": ["domain_unclear"]}]


def test_dev_check_by_counts(tmp_path):
    rel = _dev(tmp_path, ROWS)
    assert check_dev({"trained_on": rel, "n": 2, "n_out": 1}, tmp_path)["status"] == "ok"
    assert check_dev({"trained_on": rel, "n": 3, "n_out": 1}, tmp_path)["status"] == "mismatch"


def test_dev_check_by_sha(tmp_path):
    import hashlib

    rel = _dev(tmp_path, ROWS)
    sha = hashlib.sha256((tmp_path / rel).read_bytes()).hexdigest()
    assert check_dev({"trained_on": rel, "dev_sha256": sha}, tmp_path)["status"] == "ok"
    assert check_dev({"trained_on": rel, "dev_sha256": "0" * 64}, tmp_path)["status"] == "mismatch"


def test_dev_check_missing_file(tmp_path):
    assert check_dev({"trained_on": "nope.jsonl"}, tmp_path)["status"] == "dev_not_found"
