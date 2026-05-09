# Miniforge Agent Guidelines

This file is the root operating guide for AI agents working in this repo. `CLAUDE.md` is intentionally a symlink to this file so Claude Code and other agents share the same project rules.

## 0 - Purpose

Miniforge is a Python library for MiniMax M2.7 inference, optimized for GMKtech M7 local hardware.

These rules protect maintainability, local performance, offline-friendly development, and clear backend behavior. Rules marked **MUST** are hard requirements unless the user explicitly asks for a different tradeoff. Rules marked **SHOULD** are strong defaults.

## 1 - Project Context

- **PC-1 (MUST)** Package name is `miniforge`; source lives in `src/miniforge/`.
- **PC-2 (MUST)** Preserve public imports from `miniforge`: `Miniforge`, `InferenceEngine`, and `M7Config`.
- **PC-3 (MUST)** Preserve backend order unless asked to change it: `llama_cpp` preferred, `transformers` fallback.
- **PC-4 (MUST)** Keep configuration centered on `M7Config` and YAML files in `configs/`.
- **PC-5 (MUST)** Keep memory defaults conservative for the GMKtech M7 target.
- **PC-6 (SHOULD)** Optimize for the expected local hardware: AMD Ryzen 7 PRO 6850H, Windows 11 + WSL2.
- **PC-7 (SHOULD)** Preserve the expected runtime profile when changing inference code: Q4_K_M, turbo3 KV cache, 15-25 tok/s generation target, and roughly 4-5GB memory use where practical.

## 2 - Before Coding

- **BP-1 (MUST)** Read this file and any directory-specific `AGENTS.md` before editing files in that directory.
- **BP-2 (MUST)** Build context from existing code first; do not assume architecture from filenames alone.
- **BP-3 (MUST)** Ask one short clarifying question when requirements are ambiguous, risky, or could break existing behavior.
- **BP-4 (SHOULD)** For complex work, make the smallest viable plan before editing.
- **BP-5 (SHOULD)** If two credible approaches exist, prefer the one with fewer new abstractions, fewer touched files, and better consistency with existing code.

## 3 - While Coding

- **C-1 (MUST)** Keep all public APIs typed; mypy runs in strict mode for source.
- **C-2 (MUST)** Do not add blocking I/O inside `async def` paths.
- **C-3 (MUST)** Preserve async behavior and streaming behavior when touching generation or chat paths.
- **C-4 (MUST)** Keep backend errors clear when a backend is unavailable or fallback cannot be used.
- **C-5 (MUST)** Never hardcode secrets; use environment variables or documented config.
- **C-6 (MUST)** Keep memory limits aligned with `M7Config` semantics.
- **C-7 (SHOULD)** Change the narrowest module that owns the behavior.
- **C-8 (SHOULD)** Reuse existing config, hardware, memory, model-resolution, and generation utilities before adding new abstractions.
- **C-9 (SHOULD)** Prefer small testable functions over new classes unless the existing design calls for a class.
- **C-10 (SHOULD NOT)** Add comments unless they explain a non-obvious constraint, caveat, or hardware/backend reason.
- **C-11 (SHOULD NOT)** Add backward-compatibility code unless persisted data, shipped behavior, external users, or the user explicitly requires it.

## 4 - Testing

- **T-1 (MUST)** Add or update tests for bug fixes and public API behavior changes.
- **T-2 (MUST)** Keep tests deterministic and offline by default.
- **T-3 (MUST)** Avoid network calls, large model downloads, and hardware-specific assumptions in unit tests.
- **T-4 (MUST)** Keep assertions specific to user-visible behavior: return shape, messages, exceptions, and config resolution.
- **T-5 (SHOULD)** Prefer unit tests with fakes/mocks for model and network boundaries.
- **T-6 (SHOULD)** Use integration tests when validating config loading, backend fallback, CLI behavior, or public API wiring.
- **T-7 (SHOULD NOT)** Add brittle timing-based assertions in async tests.

## 5 - Config And Runtime Defaults

- **D-1 (MUST)** Treat values in `configs/` as user-facing defaults.
- **D-2 (MUST)** Keep config keys stable; include migration/docs if a rename is unavoidable.
- **D-3 (MUST)** Keep YAML valid, readable, and secret-free.
- **D-4 (SHOULD)** Prefer additive config changes over breaking shape changes.
- **D-5 (SHOULD)** Keep resource limits realistic for local CPU inference.
- **D-6 (SHOULD)** Preserve useful environment-variable overrides, including `MINIFORGE_MODEL`, `MINIFORGE_BACKEND`, `MINIFORGE_QUANTIZATION`, `MINIFORGE_PRESET`, `MINIFORGE_OFFLINE`, `MINIFORGE_MODEL_DIRS`, `MINIFORGE_MODEL_WEIGHTS_PATH`, `MINIFORGE_MAX_TOKENS`, and `MINIFORGE_TEMPERATURE`.

## 6 - Code Organization

- **O-1 (MUST)** Keep library code under `src/miniforge/`.
- **O-2 (MUST)** Keep examples under `examples/` simple, copy-paste friendly, and aligned with current APIs.
- **O-3 (MUST)** Keep tests under `tests/` consistent with existing pytest style and fixture patterns.
- **O-4 (SHOULD)** Update affected examples in the same task when API signatures change.
- **O-5 (SHOULD)** Keep one concept per example: chat, streaming, tools, vision, server, or CLI.

## 7 - Tooling Gates

- **G-1 (MUST)** Format Python with Black before handoff when Python files change.
- **G-2 (MUST)** Run Ruff for changed source/tests when practical.
- **G-3 (MUST)** Run mypy for public API, config, backend, or type-heavy changes.
- **G-4 (MUST)** Run targeted pytest first, then broader pytest for non-trivial changes.
- **G-5 (SHOULD)** Reinstall editable package after packaging changes.

## 8 - Git And GitHub

- **GH-1 (MUST)** Commit only when the user explicitly asks.
- **GH-2 (MUST)** Push only when the user explicitly asks, including when they invoke `qgit`.
- **GH-3 (MUST)** Use `gh` for GitHub work such as issues, PRs, checks, and releases.
- **GH-4 (SHOULD)** Use detailed, intent-focused commit messages.
- **GH-5 (SHOULD)** Use Conventional Commits when writing commit messages.
- **GH-6 (SHOULD NOT)** Mention AI tools, Claude, or Anthropic in commit messages.

## 9 - Development Commands

```bash
# Install all project extras with uv when possible
uv pip install -e ".[all]"

# Install llama.cpp backend only; Windows requires a C++ toolchain or matching wheel
uv pip install -e ".[llama-cpp]"

# Validation order
uv black src/ tests/
uv ruff check src/ tests/
uv mypy src/
uv pytest tests/test_core.py   # quick engine/config/backend check
uv pytest tests/test_tools.py  # tool calling check
uv pytest                      # full suite; slower

# Reinstall after packaging changes
python -m pip install -e .
```

## 10 - Tooling Config

- **Black:** line length 100, targets Python 3.10, 3.11, and 3.12.
- **Ruff:** line length 100, target Python 3.10, enabled rule groups `E`, `W`, `F`, `I`, `N`, `UP`, `B`, `C4`, and `SIM`.
- **Ruff ignores:** `E501` because Black handles line length, plus `B008`, `C901`, and `N818`.
- **Mypy:** Python 3.10, `check_untyped_defs`, `disallow_untyped_defs`, `disallow_incomplete_defs`, `warn_redundant_casts`, `warn_unused_ignores`, `warn_return_any`, and `strict_equality`.
- **Pytest:** `asyncio_mode = auto`, tests under `tests`, files named `test_*.py`, addopts `-v --tb=short`.

## 11 - Directory Rules

- `src/AGENTS.md` applies to `src/miniforge/` and covers public API preservation, async consistency, backend fallback, and source validation.
- `tests/AGENTS.md` applies to `tests/` and covers deterministic offline tests, pytest style, and execution expectations.
- `examples/AGENTS.md` applies to `examples/` and covers copy-paste friendly examples with explicit runtime assumptions.
- `configs/AGENTS.md` applies to `configs/` and covers stable config keys, secret-free YAML, and realistic resource limits.

## 12 - Function Quality Checklist

Use this checklist for every major function you add or edit:

1. Can a maintainer read the function and honestly follow the control flow without comments?
2. Is the cyclomatic complexity low enough for safe review and testing?
3. Would a standard data structure or simpler algorithm make the function clearer?
4. Are all parameters used and necessary?
5. Are casts, ignores, or dynamic typing avoided unless they isolate a real boundary?
6. Is the function testable without downloading models, contacting networks, or requiring special hardware?
7. Are hidden dependencies passed in or covered by integration tests when they can affect behavior?
8. Does the name match existing Miniforge domain vocabulary?
9. Is extraction justified by reuse, testability, or a clear readability win?

## 13 - Test Quality Checklist

Use this checklist for every major test you add or edit:

1. Can the test fail for a real defect?
2. Does the test name describe the behavior verified by the final assertion?
3. Are expectations independent of the implementation under test?
4. Are edge cases, realistic inputs, unexpected inputs, and boundaries covered where relevant?
5. Are assertions strong and specific rather than broad or incidental?
6. Does the test avoid network, model downloads, timing fragility, and local hardware assumptions?
7. Does the test follow the same formatting, typing, and style rules as production code?
8. Does the test avoid checking behavior already guaranteed only by the type checker?

## 14 - Shortcuts

The user may invoke these shortcuts at any time.

### qnew

Read and follow all best practices in this file and any applicable directory-specific `AGENTS.md` files.

### qplan

Analyze similar code before planning. Confirm the plan is consistent with the repo, introduces minimal changes, reuses existing utilities, and preserves async/backend behavior.

### qcode

Implement the agreed plan. Add or update tests for changed behavior. Run formatting, linting, type checking, and targeted tests first; run the full suite for broad or risky changes.

### qcheck

Act as a skeptical senior engineer. Review every major change against the implementation rules, function quality checklist, test quality checklist, backend fallback behavior, async safety, and memory constraints.

### qcheckf

Act as a skeptical senior engineer. Review every major function added or edited against the function quality checklist.

### qcheckt

Act as a skeptical senior engineer. Review every major test added or edited against the test quality checklist.

### qux

Imagine you are a human tester of the changed feature. Output prioritized manual test scenarios, with highest-risk paths first.

### qgit

If there are changes worth committing, add relevant files, create a detailed Conventional Commit, and push to the remote. Use `gh` for GitHub operations when needed. Do not include secrets or unrelated files.
