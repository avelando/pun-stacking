import pytest

from pun_stacking.pipeline import STAGES, run_pipeline, run_stage


def test_stage_names_are_unique():
    names = [stage.name for stage in STAGES]

    assert len(names) == len(set(names))


def test_expected_stages_exist():
    names = {stage.name for stage in STAGES}

    assert names == {
        "data",
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


def test_pipeline_stops_on_unimplemented_stage():
    with pytest.raises(
        NotImplementedError,
        match="Dataset preparation",
    ):
        run_pipeline()