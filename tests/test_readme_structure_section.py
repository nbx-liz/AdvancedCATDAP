from pathlib import Path


def test_project_structure_section_has_no_mojibake_marker():
    content = Path("README.md").read_text(encoding="utf-8")
    assert "## Project Structure" in content
    assert "隨" not in content
