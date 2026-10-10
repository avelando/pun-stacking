from pathlib import Path

import pytest

from pun_stacking.data.audit import load_jsonl
from pun_stacking.data.folds import create_group_folds


def make_examples(n_pairs: int = 25) -> list[dict]:
    rows = []

    for index in range(n_pairs):
        group = f"1.{index}"

        rows.extend(
            [
                {
                    "id": f"{group}.H",
                    "text": f"Pun {index}",
                    "label": 1,
                },
                {
                    "id": f"{group}.N",
                    "text": f"Non-pun {index}",
                    "label": 0,
                },
            ]
        )

    return rows


def test_every_pair_is_assigned():
    assignments, summaries = create_group_folds(
        make_examples(),
        n_splits=5,
        random_state=42,
    )

    assert len(assignments) == 25
    assert set(assignments.values()) == set(range(5))
    assert len(summaries) == 5


def test_folds_are_balanced():
    _, summaries = create_group_folds(
        make_examples(),
        n_splits=5,
        random_state=42,
    )

    for summary in summaries:
        assert summary["train_pairs"] == 20
        assert summary["validation_pairs"] == 5
        assert summary["train_examples"] == 40
        assert summary["validation_examples"] == 10
        assert summary["validation_class_counts"] == {
            "0": 5,
            "1": 5,
        }


def test_folds_are_deterministic():
    rows = make_examples()

    first, _ = create_group_folds(rows, 5, 42)
    second, _ = create_group_folds(rows, 5, 42)

    assert first == second


def test_incomplete_pair_is_rejected():
    rows = make_examples()
    rows.pop()

    with pytest.raises(
        ValueError,
        match="Every pair must contain",
    ):
        create_group_folds(rows, 5, 42)


def test_real_pair_controlled_folds():
    root = Path(__file__).resolve().parents[1]
    path = root / "data" / "pair_controlled" / "seed_13" / "train.jsonl"

    if not path.exists():
        pytest.skip("Puntuguese dataset is not installed")

    rows = load_jsonl(path)

    assignments, summaries = create_group_folds(
        rows,
        n_splits=5,
        random_state=42,
    )

    assert len(assignments) == 1995
    assert len(summaries) == 5

    for summary in summaries:
        assert summary["train_pairs"] == 1596
        assert summary["validation_pairs"] == 399
        assert summary["train_examples"] == 3192
        assert summary["validation_examples"] == 798
        assert summary["validation_class_counts"] == {
            "0": 399,
            "1": 399,
        }
