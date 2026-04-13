# Source Tree Agent Guide

Applies to everything under `src/`.

## Priorities
- Preserve importable public APIs from `miniforge`.
- Keep async behavior consistent; avoid blocking calls in async paths.
- Maintain backend fallback behavior (`llama_cpp` vs `transformers`) unless explicitly changing it.

## Implementation Guidance
- Prefer changes in the narrowest module that owns the behavior.
- Reuse existing config and utility helpers before adding new abstractions.
- Keep memory-related defaults conservative and aligned with existing config semantics.
- Raise clear errors for unsupported backends/config combinations.

## Required Validation
- Run tests most relevant to modified code:
  - `pytest tests/test_core.py`
  - `pytest tests/test_tools.py`
- For broader model/config/backend changes, run full suite: `pytest`.
