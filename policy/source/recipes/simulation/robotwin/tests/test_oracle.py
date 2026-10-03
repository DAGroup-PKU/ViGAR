import pytest

from recipes.simulation.robotwin.goalwam.oracle import load_oracle_references, validate_oracle_config


@pytest.mark.parametrize("roots", [[], "directory", [""], ["one", "one"]])
def test_invalid_oracle_roots_fail_early(roots):
    with pytest.raises(ValueError, match="oracle_reference_roots"):
        validate_oracle_config({"oracle_reference_roots": roots})


def test_missing_oracle_does_not_silently_run_a_different_seed(tmp_path):
    with pytest.raises(TimeoutError, match="No completed oracle reference"):
        load_oracle_references(dict(oracle_reference_roots=[str(tmp_path)], oracle_reference_timeout=0.01), "task")
