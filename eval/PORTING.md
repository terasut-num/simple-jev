# Evaluation tooling provenance and scope

This directory consolidates the endpoint evaluation framework previously developed
outside main. It does not contain a new model evaluation or change model weights,
server prompts, scoring policy selection, or inference performance.

## Sources

- Framework, 172 categorized suite manifests, preparation converters, native
  adapters, taxonomy, tests, licenses, source commitments, and compact historical
  Jev benchmark reports: `simple-jev-eval`, base revision
  `c80011cc6a9100ee1f8ec2f01b47d1aff63876ee`.
- The reviewed working-tree additions at migration time were also included:
  native public JevBench adapter/tests/manifest, the 231-row public fixture,
  pinned upstream scorer, its source manifest, and the published outcome reference.
  Those additions were not all in that source revision. Upstream checksums and
  revision `83831807458d7df424a1e53e5724f3a3ffe2cf89` are recorded in
  `vendor/jevbench/source-manifest.json`; offline tests verify fixture/scorer hashes.
- `full-suites.json` is copied unchanged from
  `simple-jev-prompt-lab/experiments/full_eval/suites.json`, source worktree based
  on revision `008bff4dba092be033f4032945c8c6179ee2d679`. It freezes 65 text and
  seven image suites, rather than scanning the catalog and double-counting splits.
- `audit.py` and `compare.py` adapt the completion-replay and matched-item reporting
  procedures from that worktree's full-evaluation experiment into standalone
  endpoint-artifact tools. They have no cloud authentication, job-state, internal
  filesystem, GPU runtime, or experiment-hook dependency.

Upstream preparation sources and licenses remain in `vendor/`; the surrounding
repository's license does not replace third-party dataset terms. Large datasets,
private source records, downloaded images, weights, and raw model predictions
are not copied. Small existing distributable fixtures and compact historical
reports retain their provenance. Historical raw-archive paths in those reports
are evidence identifiers, not prerequisites for new evaluations.

## Migration changes

- Added portable quick/decision/full-text/vision/full presets, a no-network
  listing command, optional user-supplied deployment metadata, offline raw-response
  replay, and explicit comparison modes matching the public page's weighting.
- Kept the original top-level SemIf manifest paths as catalog aliases. Minimal
  custom manifests remain runnable; absent taxonomy is labeled `unspecified`.
- Added reporting metadata to native public JevBench, keeping its upstream scoring
  separate from the older typed JevBench full-suite adapter. No metric substitution
  or change to the historical reference scores.
- New run manifests commit evaluator/scorer dependencies, including the vendored
  native JevBench scorer. Reporting, repair and resume reject changed dependencies.
  Legacy archives retain their original, narrower selected-adapter commitments.
- Fixed image repair to persist/rebind asset roots and revalidate image bytes before
  retries. Repair now also permits unauthenticated local endpoints.
- Fixed the missing closing brace in main's old Choice request builder by porting
  the working evaluator implementation. The generic validator's existing support
  for up to 255 options is retained; this does **not** increase any server's
  candidate/context limit.
- Added standard-library migration, localhost HTTP, audit, comparison, and asset
  repair regression tests, plus an offline Python 3.10/3.12 CI workflow.

Cloud-specific launchers, cost monitors, automatic resubmission/rebalancing,
48-worker GPU sharding/recovery scripts, experimental prompt hooks, and backend
patches are intentionally not installed in the public evaluation client. Manage
servers separately; select supported startup policies explicitly and preserve
runtime provenance. This port does not submit jobs, rebuild containers, or assert
numeric equivalence across execution environments.

The website snapshot exporter remains a separate, historical-archive exporter
with its documented archive layout. For new portable runs use `report.py`,
`audit.py` and `compare.py`; regenerating the exact published website snapshot
also requires its original model predictions and images, which are not in Git.

## Migration validation

- 75 evaluation tests passed on Python 3.12.3, including an isolated copy with
  site packages disabled, localhost HTTP execution/resume, native scorer replay,
  corrupted-artifact rejection and image retry validation. CI for 3.10/3.12 is
  configured; this is not a claim that hosted CI has already run.
- In that isolated checkout, the three full JevBench tiers rebuilt from vendored
  sources, and TypeSafe rebuilt from existing verified official snapshots. The
  native quick selection validated as exactly 231 + 144 + 102 = 477 rows.
- All 20 decision preset suites validated against existing prepared source data:
  32,865 examples / 44,600 labels executed, including 11,501 additional CodeMMLU
  knowledge examples needed for whole-project coverage. The comparison selects
  exactly the original 26 decision items (21,364 examples / 33,099 labels).
- Recomputed comparison outputs from the five saved model reports exactly match
  the published rounded decision scores (89.87, 88.60, 78.71, 88.77, 83.37%) and
  vision accuracy means (86.52, 88.29, 85.62, 86.14, 83.00%), in Qwen27B, QwenMoE,
  Qwen4B, GemmaMoE, Gemma12B order. This is report-level verification, not fresh
  inference or a new raw-response audit of those historical archives.
- All 32 existing website unit tests passed; no website score/content changes.
