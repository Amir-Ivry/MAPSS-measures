# Contributing

Thanks for helping improve MAPSS-measures!

## Quick Start
1. Fork the repo and create a feature branch.
2. Create a fresh Python 3.11 env (Conda or venv).
3. Install deps:
   ```bash
   pip install -r requirements.txt
   pip install -r dev-requirements.txt
   ```
4. Run lint & tests:
   ```bash
   ruff check .
   black --check .
   pytest -q
   ```
5. Submit a PR describing your change.

## Code Style
- Use **Black** for formatting.
- Use **Ruff** for linting.
- Keep functions small and focused.
- Avoid broad `except:` blocks; log exceptions.

## Tests
- Put tests in `tests/`.
- Keep them small and fast.
- For GPU-dependent code, tests should fall back to CPU if CUDA is unavailable.
