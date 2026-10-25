"""Runner/report integration with cheap fixed models; not performance tests."""

import pytest

from sportka.features import generate_synthetic_data
from sportka.models import RandomPredictor


class UniformModel(RandomPredictor):
    def __init__(self, **kwargs):
        pass


@pytest.mark.parametrize("version", ["v1", "v2", "null"])
def test_runners_use_all_heldout_targets_and_write_report(tmp_path, monkeypatch, version):
    from sportka.experiments import run_experiment as v1, run_experiment_v2 as v2, run_null_test as null
    module = {"v1": v1, "v2": v2, "null": null}[version]
    monkeypatch.setattr(module, "_REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(module, "N_BOOTSTRAP", 20)
    monkeypatch.setattr(module, "generate_synthetic_data", lambda n_draws, seed: generate_synthetic_data(40, seed))
    if version == "v1":
        monkeypatch.setattr(module, "MODELS", {name: UniformModel() for name in module.MODELS})
        results = module.run_experiment()
    elif version == "v2":
        for name in ("UBTModel", "UBTMLPV2", "UBTCNNv2"):
            monkeypatch.setattr(module, name, UniformModel)
        results = module.run_experiment()
    else:
        monkeypatch.setattr(module, "UBTMLPV2", UniformModel)
        results = module.run_null_test()
    for result in results.values():
        assert len(result["Y_test"]) == 6
        assert (result["Y_test"].sum(axis=1) == 6).all()
    reports = list((tmp_path / "reports").glob("*.md"))
    assert reports
    assert "conditional_set_nll" in reports[0].read_text()


def test_null_screen_never_labels_worse_performance_as_signal():
    from sportka.experiments.run_null_test import _apply_decision
    def result(point, low, high):
        return {"metrics": {"recall_at_6": point}, "ci": {"recall_at_6": (point, low, high)}}
    controls = {name: result(.12, .11, .13) for name in ("shuffled", "random", "reversed")}
    assert _apply_decision({"real": result(.04, .03, .05), **controls})[0] == "NO_SIGNAL"
    assert _apply_decision({"real": result(.12, .11, .13), **controls})[0] == "NO_SIGNAL"
    assert _apply_decision({"real": result(.20, .19, .21), **controls})[0] == "POTENTIAL_SIGNAL"
