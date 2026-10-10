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


def test_unimplemented_stage():
    with pytest.raises(NotImplementedError):
        run_stage("graphs")


def test_pipeline_stops_on_unimplemented_stage(monkeypatch):
    from pun_stacking import pipeline

    monkeypatch.setattr(
        pipeline,
        "STAGES",
        (
            pipeline.Stage(
                "evaluation",
                "Run final test evaluation",
                pipeline.evaluate,
            ),
        ),
    )

    with pytest.raises(
        NotImplementedError,
        match="Final test evaluation",
    ):
        pipeline.run_pipeline()
