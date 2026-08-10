from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

import mapss


ROOT = Path(__file__).resolve().parents[1]


def test_version_is_synchronized():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project_version = re.search(r'^version = "([^"]+)"$', pyproject, flags=re.MULTILINE)
    assert project_version is not None
    assert project_version.group(1) == mapss.__version__
    assert f'version: "{mapss.__version__}"' in (ROOT / "CITATION.cff").read_text(
        encoding="utf-8"
    )
    assert f"## {mapss.__version__}" in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")


def test_public_files_do_not_contain_private_machine_paths():
    public_files = [
        *ROOT.glob("*.md"),
        *ROOT.glob("docs/*.md"),
        *ROOT.glob("Manifests/*.py"),
        *ROOT.glob("Manifests/*.json"),
        *ROOT.glob("examples/*"),
    ]
    forbidden = ("C:/postdoc", "C:\\postdoc", "/home/sivry")
    for path in public_files:
        text = path.read_text(encoding="utf-8")
        assert not any(value in text for value in forbidden), path


def test_relative_markdown_links_exist():
    markdown_files = [ROOT / "README.md", *ROOT.glob("docs/*.md")]
    link_pattern = re.compile(r"\[[^]]+\]\(([^)]+)\)")
    for markdown in markdown_files:
        for raw_target in link_pattern.findall(markdown.read_text(encoding="utf-8")):
            target = raw_target.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (markdown.parent / unquote(target)).resolve()
            assert resolved.exists(), f"Broken link in {markdown}: {raw_target}"


def test_python_examples_expose_help():
    for example in ("run_from_paths.py", "smoke_test.py"):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "examples" / example), "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert "usage:" in completed.stdout.lower()
