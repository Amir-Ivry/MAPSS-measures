# Contributing

Contributions that improve correctness, reproducibility, model coverage, performance, or
documentation are welcome.

1. Open an issue describing the proposed change and its effect on PS/PM behavior.
2. Create a focused branch from `main`.
3. Install the development environment with `pip install -e ".[test]"`.
4. Add or update tests. Scientific changes must include a regression fixture or a clear
   comparison against the previously released behavior.
5. Run `ruff check .`, `pytest`, `python -m build`, and `twine check dist/*`.
6. Open a pull request using the repository template.

Changes to distortion banks, preprocessing, activity masking, diffusion-map construction,
PS/PM equations, defaults, or aggregation affect scientific comparability. Such changes must
be explicit in the changelog and use an appropriate version bump; do not present changed scores
as directly interchangeable with an earlier release.
