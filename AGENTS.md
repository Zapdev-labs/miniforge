# Miniforge Agent Guide

## Mission
- Keep Miniforge reliable for local MiniMax inference on constrained hardware.
- Prefer small, focused edits that preserve public behavior unless a change request says otherwise.

## Repo Facts
- Language: Python (`>=3.10`).
- Packaging: `pyproject.toml` with `hatchling`.
- Source root: `src/miniforge`.
- Tests: `tests/` using `pytest`.
- Typical runtime target: local CPU inference, memory-constrained systems.

## Working Rules
- Do not start development servers unless explicitly requested.
- Run builds/checks/tests instead of long-running dev commands.
- Avoid creating new markdown files unless requested; update existing docs when needed.
- Never hardcode secrets or API keys; use environment variables/config files.
- Keep comments minimal and only where logic is genuinely non-obvious.

## Code Style Expectations
- Follow existing style and structure in nearby files.
- Preserve strict typing and avoid introducing untyped public APIs.
- Keep imports at the top of files.
- Make behavior-preserving refactors unless user asks for feature changes.

## Validation Before Handoff
- Run targeted tests first, then broader suite if change scope grows:
  - `pytest tests/test_core.py`
  - `pytest tests/test_tools.py`
  - `pytest`
- If config or packaging changes are made, also run:
  - `python -m pip install -e .`

## Directory-Level Overrides
- If a subdirectory contains its own `AGENTS.md`, follow the closest one first.
