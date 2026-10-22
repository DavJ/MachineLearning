from sportka.config import Config, DEFAULT_CONFIG


def test_nested_defaults_are_independent_and_reset(tmp_path):
    a, b = Config(str(tmp_path / "a.json")), Config(str(tmp_path / "b.json"))
    original = DEFAULT_CONFIG["training"]["default_epochs"]
    a.set("training", "default_epochs", original + 1)
    assert b.get("training", "default_epochs") == original
    assert DEFAULT_CONFIG["training"]["default_epochs"] == original
    a.reset_to_defaults()
    assert a.get("training", "default_epochs") == original
    a.set("training", "default_epochs", original + 2)
    assert DEFAULT_CONFIG["training"]["default_epochs"] == original
