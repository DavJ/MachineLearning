"""Download the public Allwyn CSV export and validate it before replacement."""

import os
from pathlib import Path
import tempfile

from sportka.data import read_history

HISTORY_URL = "https://www.allwyn.cz/loterie/historie-cisel?game=sportka"


def download_data_from_sazka(directory="/tmp", timeout=30):
    """Compatibility API: explicitly fetch CSV without Chrome or Selenium.

    Validate in a temporary file, then atomically replace sportka.csv. A failed
    request or invalid export never overwrites the user's existing history.
    Allwyn currently serves the text export as application/ms-excel.
    """
    import requests

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "sportka.csv"
    response = requests.get(HISTORY_URL, timeout=timeout)
    response.raise_for_status()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".csv", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(response.content)
        read_history(temporary)
        os.replace(temporary, destination)
        return str(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
