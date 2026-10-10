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
        "graphs",
        "bertimbau",
        "embeddings",
        "base-models",
        "early-fusion",
        "stacking",
        "evaluation",
    }


def test_unknown_stage():
    with pytest.raises(ValueError, match="Unknown stage"):
        run_stage("invalid")


def test_unimplemented_stage():
    with pytest.raises(NotImplementedError):
        run_stage("graphs")


def test_pipeline_stops_on_unimplemented_stage(monkeypatch):
    from pun_stacking import pipeline

    completed_data = pipeline.Stage(
        "data",
        "Prepare and validate dataset",
        lambda: None,
    )

    completed_folds = pipeline.Stage(
        "folds",
        "Create pair-grouped OOF folds",
        lambda: None,
    )

    monkeypatch.setattr(
        pipeline,
        "STAGES",
        (
            completed_data,
            completed_folds,
            *pipeline.STAGES[2:],
        ),
    )

    with pytest.raises(
        NotImplementedError,
        match="Graph construction",
    ):
        pipeline.run_pipeline()
