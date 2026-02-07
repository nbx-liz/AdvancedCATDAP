import tomllib
from pathlib import Path


def test_optional_dependency_all_has_no_duplicates():
    pyproject = Path("pyproject.toml")
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    all_extras = data["project"]["optional-dependencies"]["all"]
    assert len(all_extras) == len(set(all_extras))
