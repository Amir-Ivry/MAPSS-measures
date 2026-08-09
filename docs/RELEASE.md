# Release checklist

## One-time PyPI setup

1. Confirm that the distribution name `mapss-measures` is available on PyPI.
2. In PyPI account publishing settings, add a pending Trusted Publisher with:
   - PyPI project: `mapss-measures`
   - GitHub owner: `Amir-Ivry`
   - repository: `MAPSS-measures`
   - workflow: `publish.yml`
   - environment: `pypi`
3. In the GitHub repository settings, create an environment named `pypi`.
4. Do not add a long-lived PyPI token. The workflow uses short-lived OIDC credentials.

## Release 1.1.0

1. Merge the package pull request only after every CI job passes.
2. Run one wav2vec 2.0 layer-2 example on a CUDA machine and retain its inputs, command,
   package/environment versions, and output CSV hashes.
3. Confirm that `pyproject.toml`, `mapss.__version__`, `CHANGELOG.md`, and
   `CITATION.cff` all say `1.1.0`.
4. Create and push the signed or annotated tag:

   ```bash
   git tag -a v1.1.0 -m "MAPSS Python package 1.1.0"
   git push origin v1.1.0
   ```

5. Watch the `Publish to PyPI` workflow. It tests, builds, and publishes through Trusted Publishing.
6. Create a GitHub release from `v1.1.0`. Use the 1.1.0 changelog and attach the preserved
   model smoke-test record.
7. Verify from a clean environment:

   ```bash
   python -m venv mapss-release-check
   source mapss-release-check/bin/activate
   pip install mapss-measures==1.1.0
   python -c "import mapss; print(mapss.__version__)"
   ```

8. Replace GitHub-only installation wording in challenge material with
   `pip install mapss-measures==1.1.0`.

## Challenge handoff package

Give organizers four links: the PyPI page, repository quick start, pinned ICLR paper, and a
frozen evaluator example. Ask them to pin version and settings rather than tracking `main`.
