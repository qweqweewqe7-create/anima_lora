# project/finished/ — completed lines

Lines that ran to a **successful conclusion** — the goal was reached or the
measured ceiling was hit — land here, one subdir each. This tier is distinct
from the gitignored `_archive/` tree on both axes:

- **Outcome**: `finished/` lines *worked* (shipped artifacts, ceiling reached);
  `_archive/` holds retired/killed/superseded material.
- **Visibility**: `finished/` is **tracked** — the record of what was achieved
  and where the ceiling sits stays in the public repo, so nobody re-proposes a
  lever that was already exhausted.

Each subdir is a digest home page (`README.md`): final verdicts with evidence
pointers, shipped artifacts, and any small open remainder. It does **not**
duplicate code or bench tables — canonical sources stay where they live.

A finished line's whole working tree may move here when its top-level surface
(make targets, repo-root dir) is no longer worth keeping — the code stays
runnable via direct script invocation. Each line's `STATUS.md` (or `README.md`
for digest-only homes) records the verdicts.

Finished lines:

- [`sr/`](sr/) — ResShift ×4/×2 super-resolution sidecar (full tree moved
  from repo-root `sr/`; `make sr-*` targets removed). ×2 and ×4 both at their
  measured ceilings (teacher-bound; distillation faithful). Open remainder:
  Korean-text training data. Verdicts: [`sr/STATUS.md`](sr/STATUS.md); ops:
  [`sr/README.md`](sr/README.md).
