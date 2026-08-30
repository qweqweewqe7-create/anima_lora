# bench/tagger_external

`run_bench.py` (in-house PE dual-encoder head vs an external dbv4 tagger) was
archived 2026-08-30 to `_archive/anima_tagger_training/pe_backend_removed_2026_08_30/bench_tagger_external/`
when the PE tagger backend was removed — "ours" *is* dbv4-backed now, so the
comparison it ran is moot. Results under `results/` predate that.

Still live: `calibration_check.py` (CPU-only ECE check on the dbv4 sidecar
probs) and `probe_position_rescore.py`.
