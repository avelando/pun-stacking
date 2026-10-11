import pytest

from pun_stacking.pipeline import STAGES, run_stage


def test_stage_names_are_unique():
    names = [stage.name for stage in STAGES]

    assert len(names) == len(set(names))


def test_expected_stages_exist():
    names = {stage.name for stage in STAGES}

    assert names == {
        "data",
        "folds",
        "historical-baseline",
        "graphs",
        "bertimbau",
        "embeddings",
        "base-models",
        "early-fusion",
        "stacking",
        "validation-report",
        "evaluation",
    }


def test_unimplemented_stage():
    with pytest.raises(
        NotImplementedError,
        match="Final test evaluation",
    ):
        run_stage("evaluation")


def test_full_pipeline_excludes_protected_stages(monkeypatch):
    from pun_stacking import pipeline

    executed = []

    monkeypatch.setattr(
        pipeline,
        "STAGES",
        (
            pipeline.Stage(
                "data",
                "Prepare dataset",
                lambda: executed.append("data"),
            ),
            pipeline.Stage(
                "freeze-protocol",
                "Freeze protocol",
                lambda: executed.append("freeze"),
            ),
            pipeline.Stage(
                "evaluation",
                "Final evaluation",
                lambda: executed.append("evaluation"),
            ),
        ),
    )

    pipeline.run_pipeline()

    assert executed == ["data"]
