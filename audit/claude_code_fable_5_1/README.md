# audit/claude_code_fable_5_1

Independent audit of `diffpfa` (commit `b1717dd`) by Claude Code running Claude Fable 5.1, 2026-09-05.

**Start with `REPORT.md`.** Everything else here is the evidence behind it.

- `REPORT.md` — findings, verified-correct behaviour, drop-in fixes, vendor reverse-engineering, recommendations.
- `a01` … `a13` — one script per measurement; each has a docstring saying what it tests and how to run it. Run from the repository root with `/home/feildaw/mypyenv/bin/python audit/claude_code_fable_5_1/aNN_*.py`.
- `proposed_tests/test_diffpfa_audit.py` — 12 pytest tests; 5 lock in verified behaviour, 7 document defects and fail on the current code.
- `out/` — JSON results, sicdcheck logs, corrected XML, the fresh SLANT and GROUND products for 2023-09-11, and one comparison figure.

Nothing under `diffpfa/`, `tests/`, `simulation/` or `tools/` was modified. `PI/` was not read.
