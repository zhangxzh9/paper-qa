from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
from dotenv import load_dotenv
from lmi.utils import (
    ANTHROPIC_API_KEY_HEADER,
    CROSSREF_KEY_HEADER,
    OPENAI_API_KEY_HEADER,
    SEMANTIC_SCHOLAR_KEY_HEADER,
    update_litellm_max_callbacks,
)

if TYPE_CHECKING:
    from paperqa.settings import Settings
    from paperqa.types import PQASession

TESTS_DIR = Path(__file__).parent
CASSETTES_DIR = TESTS_DIR / "cassettes"


@pytest.fixture(autouse=True, scope="session")
def _load_env() -> None:
    load_dotenv()


@pytest.fixture(autouse=True, scope="session")
def _setup_default_logs() -> None:
    # Lazily import from paperqa so typeguard doesn't throw:
    # > /path/to/.venv/lib/python3.12/site-packages/typeguard/_pytest_plugin.py:93:
    # > InstrumentationWarning: typeguard cannot check these packages because they
    # > are already imported: paperqa
    from paperqa.settings import ParsingSettings
    from paperqa.utils import setup_default_logs

    setup_default_logs()
    ParsingSettings.model_fields["configure_pdf_parser"].default()


@pytest.fixture(autouse=True, scope="session")
def _defeat_litellm_callbacks() -> None:
    update_litellm_max_callbacks()


@pytest.fixture(scope="session", name="vcr_config")
def fixture_vcr_config() -> dict[str, Any]:
    return {
        "filter_headers": [
            CROSSREF_KEY_HEADER,
            SEMANTIC_SCHOLAR_KEY_HEADER,
            OPENAI_API_KEY_HEADER,
            ANTHROPIC_API_KEY_HEADER,
            "cookie",
        ],
        "record_mode": "once",
        "allow_playback_repeats": True,
        "cassette_library_dir": str(CASSETTES_DIR),
    }


@pytest.fixture(name="tmp_path_cleanup")
def fixture_tmp_path_cleanup(tmp_path: Path) -> Iterator[Path]:
    yield tmp_path
    # Cleanup after the test
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.fixture(name="agent_home_dir")
def fixture_agent_home_dir(
    tmp_path_cleanup: str | os.PathLike,
) -> Iterator[str | os.PathLike]:
    """Set up a unique temporary folder for the agent module."""
    with patch.dict("os.environ", {"PQA_HOME": str(tmp_path_cleanup)}):
        yield tmp_path_cleanup


@pytest.fixture(name="agent_index_dir")
def fixture_agent_index_dir(agent_home_dir: Path) -> Path:
    return agent_home_dir / ".pqa" / "indexes"


@pytest.fixture(scope="session", name="stub_data_dir")
def fixture_stub_data_dir() -> Path:
    return Path(__file__).parent / "stub_data"


@pytest.fixture
def agent_test_settings(agent_index_dir: Path, stub_data_dir: Path) -> Settings:
    # Lazily import from paperqa so typeguard doesn't throw:
    # > /path/to/.venv/lib/python3.12/site-packages/typeguard/_pytest_plugin.py:93:
    # > InstrumentationWarning: typeguard cannot check these packages because they
    # > are already imported: paperqa
    from paperqa.settings import Settings

    # NOTE: originally here we had usage of embedding="sparse", but this was
    # shown to be too crappy of an embedding to get past the Obama article
    settings = Settings()
    settings.agent.index.paper_directory = stub_data_dir
    settings.agent.index.index_directory = agent_index_dir
    settings.agent.search_count = 2
    settings.answer.answer_max_sources = 2
    settings.answer.evidence_k = 10
    return settings


@pytest.fixture
def agent_stub_session() -> PQASession:
    # Lazily import from paperqa so typeguard doesn't throw:
    # > /path/to/.venv/lib/python3.12/site-packages/typeguard/_pytest_plugin.py:93:
    # > InstrumentationWarning: typeguard cannot check these packages because they
    # > are already imported: paperqa
    from paperqa.types import PQASession

    return PQASession(question="What is is a self-explanatory model?")


@pytest.fixture
def stub_data_dir_w_near_dupes(stub_data_dir: Path, tmp_path: Path) -> Iterator[Path]:

    # add some near duplicate files then removes them after testing
    for filename in ("bates.txt", "obama.txt"):
        if not (tmp_path / f"{filename}_modified.txt").exists():
            with (stub_data_dir / filename).open() as f:
                content = f.read()
            with (tmp_path / f"{Path(filename).stem}_modified.txt").open("w") as f:
                f.write(content)
                f.write("## MODIFIED FOR HASH")

    yield tmp_path

    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)
