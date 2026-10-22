"""A failed or changed upstream export must not destroy local history."""

from types import SimpleNamespace

import pytest
import requests

from sportka.download import HISTORY_URL, download_data_from_sazka
from tests.test_sportka_data import HEADER, ROW


def test_public_csv_download_is_validated_and_replaces_atomically(tmp_path, monkeypatch):
    def get(url, timeout):
        assert url == HISTORY_URL
        assert timeout == 30
        return SimpleNamespace(content=(HEADER + ROW).encode(), raise_for_status=lambda: None)
    monkeypatch.setattr(requests, "get", get)
    path = download_data_from_sazka(tmp_path)
    assert path == str(tmp_path / "sportka.csv")
    assert (tmp_path / "sportka.csv").read_text() == HEADER + ROW
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize("failure", ["invalid_csv", "http_error"])
def test_bad_download_preserves_existing_file(tmp_path, monkeypatch, failure):
    destination = tmp_path / "sportka.csv"
    destination.write_text("existing history")
    def get(*args, **kwargs):
        if failure == "http_error":
            raise requests.HTTPError("503")
        return SimpleNamespace(content=b"<html>maintenance</html>", raise_for_status=lambda: None)
    monkeypatch.setattr(requests, "get", get)
    with pytest.raises((ValueError, requests.HTTPError)):
        download_data_from_sazka(tmp_path)
    assert destination.read_text() == "existing history"
    assert list(tmp_path.iterdir()) == [destination]
