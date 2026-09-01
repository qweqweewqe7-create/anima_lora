# Daemon — known issues & friction log

Running list of daemon bugs and UX friction, collected from real sessions.
Each entry: symptom → repro → suggestion. Severity: **bug** (wrong/broken),
**friction** (works but fights the user), **idea** (missing capability).

Last updated: 2026-09-01 (cjk_unmask A/B session).

---

## 1. `make run-status` is blind to daemon-launched train runs — **bug**

The default launch path now routes every train job through the daemon, and a
daemon job writes its progress stream to
`output/daemon/jobs/<id>/progress.jsonl` (the record's `progress_path`).
`scripts/run_status.py` only scans `output/logs/`, where the run dir (e.g.
`output/logs/cjk_unmask_a_20260901-1802/`) holds the snapshot TOML and the TB
events but **no `progress.jsonl`** — so `make run-status RUN=<name>` exits 1
with "no progress stream" while the run is training fine.

- Repro (2026-09-01): `make lora-gui CUSTOM=cjk_unmask_a.toml ARGS="--queue"`,
  wait until `daemon-status` shows `global_step` advancing, then
  `make run-status RUN=cjk_unmask_a` → error, every time.
- Effect: the documented "where is this run at?" front door fails for the
  *default* launch mode; you must fall back to
  `make daemon-status ARGS="--job <id>"` and read the raw `latest` event.
- Suggestion (either side works):
  - `run_status.py`: also scan `output/daemon/jobs/*/progress.jsonl` and match
    on the run/output name recorded in the stream or the job record; or
  - daemon runner: tee/symlink `progress.jsonl` into the run's
    `output/logs/<run>/` dir so the existing scanner finds it.

## 2. Two `--label`s in `daemon-run`; the job record usually ends up unlabeled — **friction**

`daemon-run`'s own `--label` must come **before** the script path; anything
after it goes to the child. Bench scripts take their own `--label`, which is
where muscle memory puts it — result: the *bench run dir* is labeled but the
*daemon job record* is not. `daemon-status` listings then show a column of
identical `bench/cjk_adapter/run_bench.py` rows with empty labels (see the
2026-09-01 grid session: ~10 unlabeled rows distinguishable only by timestamp).

- Suggestion: when the daemon-side label is absent, scan the child argv for a
  `--label <value>` and copy it into the record (display-only); or print a
  one-line hint at submit time when the child argv contains `--label` but the
  job has none.

## 3. Compact `daemon-status` rows omit the exit code — **friction**

The compact summaries show `state` but not `returncode`, so a `done` row and a
"done but rc=1" row look identical; you must open `--job <id>` per job to tell
them apart. Batch workflows (queue N, check later) want failure visibility in
the listing.

- Suggestion: add `rc=<n>` (or `state: failed`) to compact rows for terminal
  jobs; `failed` state may already cover the common case — the gap is jobs
  that exit nonzero but are recorded `done`, and signal deaths.

## 4. Submission prints 5 lines of boilerplate per job — **nit**

`_print_queued` emits the attach/kill/terminate cheat-sheet on every submit.
Queueing several jobs in a loop buries the one informative line
(`queued job <id>`) under repeated hints; a user watching the scrollback asked
"what did the queue just get?" mid-session (2026-09-01).

- Suggestion: print the cheat-sheet once per process (or only when a TTY),
  keep repeat submits to the single `queued job <id> (<desc>)` line.

## 5. `daemon-wait` has no max-wait / summary-on-timeout — **idea**

`daemon-wait` blocks until terminal. For agent/scripted use a
`--timeout <s>` that exits with a distinct code and prints the latest progress
event would avoid the caller having to wrap it in its own timeout and lose the
in-flight status (observed: harness kills the wait at its own limit and the
partial output is empty because attach buffers).

## 6. Standing issues (recorded in project memory, kept here for one place)

- **`daemon-pause` unreliable in practice** — docs describe the SIGSTOP
  tree-freeze as safe/instant, but in practice pausing live train jobs has
  burned runs; policy has been "don't risk live jobs on it". Needs a
  root-cause pass before it's trusted (interaction with CUDA/NCCL watchdogs
  suspected).
- **Stall watchdog kills first-run HF downloads** — a long quiet
  `hf_hub_download` inside a job trips the stall timeout and the job is
  killed mid-download; workaround is `--stall-timeout 0` on any job that may
  fetch models. Suggestion: treat growing files under `~/.cache/huggingface`
  (or child net I/O) as liveness, or default the watchdog off for `download`
  targets.
