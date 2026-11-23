"""A short pytest temp root on Windows.

The tracking module refuses an event file path of 240 characters or more (Windows MAX_PATH). Under pytest's default
Windows temp root (``%TEMP%\\pytest-of-<user>\\pytest-<n>\\<test name>``) the run units that several tests record
reach that limit, so without ``--basetemp`` the tests use ``%TEMP%\\nk<random>`` (or ``NNKNN_TEST_TMP`` as its parent),
removed again when the session ends. An explicit ``--basetemp`` is kept.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

_created = None


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    global _created
    if sys.platform != "win32" or config.option.basetemp:
        return
    _created = Path(tempfile.mkdtemp(prefix="nk", dir=os.environ.get("NNKNN_TEST_TMP") or tempfile.gettempdir()))
    config.option.basetemp = str(_created)
    factory = getattr(config, "_tmp_path_factory", None)
    if factory is not None:  # created before this hook ran: give it the short root too
        factory._given_basetemp = _created


def pytest_unconfigure(config):
    if _created is not None:
        shutil.rmtree(_created, ignore_errors=True)
