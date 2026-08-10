# Release checklist

MAPSS publishes from signed or annotated `v*` tags through GitHub Actions and PyPI Trusted
Publishing. Do not use a long-lived PyPI token.

## Before tagging

1. Merge only after every CI job passes.
2. Confirm that `pyproject.toml`, `mapss.__version__`, `CHANGELOG.md`, and `CITATION.cff`
   contain the same version.
3. Run unit, integration, example, plotting, build, and `twine check` validation.
4. Install the built wheel in a fresh environment and verify both CLI help commands.
5. For scientific changes, run the pretrained paper configuration and preserve inputs,
   arguments, environment versions, and result hashes.

## Release 1.1.1

```bash
git tag -a v1.1.1 -m "MAPSS practitioner release 1.1.1"
git push origin v1.1.1
```

The tag workflow tests the tagged commit, builds the wheel and source distribution,
publishes them to PyPI, and creates a GitHub release with both artifacts.

If the GitHub CLI is unavailable, push a temporary release branch instead:

```bash
git push origin main:release-v1.1.1
```

GitHub Actions checks that the branch contents and version exactly match `main`, creates the
annotated tag on the `main` commit, and runs the same tested build and publishing jobs. Delete
the temporary branch after the release completes.

## Post-release verification

Use an empty directory and environment:

```bash
python -m venv mapss-release-check
source mapss-release-check/bin/activate
python -m pip install "mapss-measures[plot]==1.1.1"
python -m pip check
python -c "import mapss; print(mapss.__version__)"
python -m mapss --help
python -m mapss.plotting --help
```

Then run one ordered-path evaluation with the default pretrained model, save the CSV files,
and create the PS/PM plot. Confirm the PyPI project and GitHub release pages show version
1.1.1 before updating challenge material.
