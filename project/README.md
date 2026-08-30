# project/ — active promoted lines

One subdir per research line that has graduated past "proposal + bench report"
into an ongoing project with open phases. Each subdir is the line's home page:

| File | Contents |
|---|---|
| `methods.md` | The implementation — what code exists, where it lives, how to run it |
| `bench.md` | Digest of measured results — omitted when the line's bench lives in-tree (its own `report.md` serves directly) |
| `questions.md` | Open questions the line has not answered |
| `roadmap.md` | Remaining phases, gates, and kill criteria |
| `outcomes.md` | Shippable/practical artifacts the line produced (optional — appears once something is ship-shaped) |

Canonical sources these digest (never duplicated wholesale):
the line's proposal(s) (frozen designs) and its bench (`report.md` = raw
verdicts + full tables, `results/` = run envelopes). A promoted line may
adopt these into its home — `project/<line>/bench/` for the bench and e.g.
`initial_proposal.md` for the founding proposal (the archived directedit_ec
and sigma_lowres lines did both); lines that haven't keep them in
`bench/<line>/` and `docs/proposal/<line>*.md`.

A line leaves the active set one of two ways:

- **Finished** — it ran to a successful conclusion (goal reached or measured
  ceiling hit). Its digest home moves to the tracked
  [`finished/`](finished/) tier so the verdicts stay visible in the repo;
  any still-operational working tree (code, make targets) stays where it is.
- **Retired** — killed, superseded, or shelved. It moves to the gitignored
  `_archive/` tree (local + preserved in the private mirror).

Retired lines so far:

- `sigma_lowres` — archived 2026-08-19 → `_archive/sigma_lowres/`. The research
  branches + paper drafts were already mirrored to the private repo
  (2026-08-15) and deleted from public origin; the shipped `--sigma_lowres`
  feature stays live (`docs/optimizations/sigma_lowres.md`).
- `directedit_ec` — archived 2026-08-19 → `_archive/directedit_ec/`. Private
  mirroring still pending; the state is snapshot in the mirror's `main` and
  in origin history. EasyEdit ship proposal + paper prep remain the owed
  write-ups if the line reopens.

Finished lines are listed in [`finished/README.md`](finished/README.md)
(the ResShift SR sidecar, 2026-08-22; mod guidance, 2026-08-24).

Active projects:

- [`cjk_aware_anima/`](cjk_aware_anima/) — native JA prompt conditioning
  via an extended T5-side vocab distilled against the EN-translation teacher.
  Encoder-side research is exhausted (rows-only design settled; rare kanji
  character names fail under every measured lever — data, objective, adapter
  capacity); **v1 ships `cjk_vocab_pack_synthja` once the glossary is signed
  off**, and the one open research lever is a real co-occurrence corpus.
  Home is three digests: [`findings.md`](cjk_aware_anima/findings.md)
  (every settled verdict + ruled-out directions),
  [`deliverables.md`](cjk_aware_anima/deliverables.md) (code, data builders,
  packs, ship contract, blocker), [`plan.md`](cjk_aware_anima/plan.md) (Phase 3
  ship → corpus extension → deferred glyph line). Measured tables stay in the
  dated reports (`reports/0816_phase2.md`, `reports/0827_names_synth.md`,
  `reports/0830_adapter_lora.md`), [`datasets/README.md`](cjk_aware_anima/datasets/README.md)
  and `bench/cjk_{adapter,distill}/results/`.
