import json
from pathlib import Path

import pytest

from pun_stacking.data.audit import (
    audit_data,
    load_jsonl,
    pair_id,
)


def test_pair_id():
    assert pair_id("1.3.H") == "1.3"
    assert pair_id("1.3.N") == "1.3"


@pytest.mark.parametrize(
    "example_id",
    ["invalid", "1.3.X", ".H", "1.3"],
)
def test_invalid_pair_id(example_id):
    with pytest.raises(ValueError):
        pair_id(example_id)


def test_invalid_label(tmp_path):
    path = tmp_path / "invalid.jsonl"

    record = {
        "id": "1.3.H",
        "text": "Example text",
        "label": 0,
    }

    path.write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid label"):
        load_jsonl(path)


def test_inconsistent_token_annotations(tmp_path):
    path = tmp_path / "invalid_tokens.jsonl"

    record = {
        "id": "1.3.H",
        "text": "Example text",
        "label": 1,
        "tokens": ["Example", "text"],
        "labels": [1],
    }

    path.write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Inconsistent token annotations",
    ):
        load_jsonl(path)


def test_real_pair_controlled_splits():
    root = Path(__file__).resolve().parents[1]

    expected_path = root / "data" / "pair_controlled" / "seed_13" / "train.jsonl"

    if not expected_path.exists():
        pytest.skip("Puntuguese splits are not installed")

    report_path = audit_data(root)

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert len(report["partitions"]) == 6

    for partition in report["partitions"]:
        assert partition["examples"] == 5700
        assert partition["pairs"] == 2850
        assert partition["cross_split_pairs"] == 0
