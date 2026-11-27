# freeze delta v0.10 → v0.15 (2026-11-26)

Baseline: results_v11/freeze_manifest_baseline_v0.10.md5 (never rewritten).
Identical files: 269 / 272
Changed files: 3
Missing files: 0

## Changed (documented, spec-mandated)
- **README.md**
  reason: spec §11: root README rewritten per stage to ACTUAL v0.15 capabilities (mandatory).
  old md5 b03839dcbb8aa309e0e2b53052ff0481
  new md5 190aca4c6cae089da4cd07611d714eb6
- **pyproject.toml**
  reason: spec Phase D: version 0.15.0, packages v8…v15 + rasm_cim, rasm-cim entry script, numpy (frozen v0.1 import reality) + scikit-learn (frozen v0.5 test infra) declared.
  old md5 3f017453ba617cd1909fb07eea97b3fa
  new md5 a66a96c9a184b3fb8050fda8ebd52bf0 (v0.15 packaging)
  new md5 18048e5fc1724e7db75c8b9e6579652e (verification cycle: SPDX license string,
          data dirs as explicit namespace packages — build warnings eliminated)
- **requirements-dev.txt**
  reason: fresh-environment reproducibility discovered: frozen v0.5 tests require scikit-learn
                      (rasm_cim_v5/learn5.py) — missing from the original extras list. Recorded as dev
                      infra; v0.1/v0.5 semantics untouched. Exact cause: fresh sandbox pip lacked it,
                      collection error "No module named 'sklearn'".
  (this entry IS documented; the UNDOCUMENTED tag below was a report-generator artifact)
  old md5 d429bfc7cf830984aa939ed9289384f2
  new md5 3d2f3e46a07f6bfab76ab2589a15d8f8

## v0.15-layer fix (not a frozen file, current development version)
- **rasm_cim_v15/artifacts.py** — `LearnedArtifacts.save/current/rollback` ignored their
  `out_dir` argument when writing/reading `current.json` (pointer was hardcoded to the repo's
  `ARTIFACTS_DIR`). Consequence: every test/training run leaked a dangling `lv_XXX`
  pointer into the tracked `rasm_cim_v15/learned/current.json` while the artifact file
  itself went to `out_dir`. Detected twice via freeze/status checks (pointer drifted to
  nonexistent lv_001). Fixed (pointer is now `<out_dir>/current.json` everywhere) and a
  regression test was added (tests_v15/test_v15.py::TestArtifacts::test_save_does_not_touch_repo_pointer).
  `rasm_cim_v15/learned/current.json` restored to the committed lv_000 pointer; full suite
  re-verified (332 passed).

## Auto-repaired this phase (recorded honestly)
- **rasm_cim/__init__.py** (frozen v0.1): briefly overwritten by the Phase C unified-CLI skeleton, detected by the freeze check at Phase D, restored byte-identical via `git checkout HEAD -- rasm_cim/__init__.py`; md5 re-verified against baseline (9c19f403e3e5f9d2f1d18739373b6233). Unified CLI remained functional because it only ADDS __main__.py/cli.py; v0.1 exports untouched after restore.

## Rust v0.7
sha256-of-sorted-md5s of rasm_cim_rust/*.rs = 451bbfa52e67a72b04c0046d9df4e9d5 — unchanged across v0.11…v0.15.

## Appendix — verification cycle amendments (2026-11-27)
- pyproject.toml md5 evolved: 3f017453… (frozen baseline) → a66a96c9… (v0.15 packaging)
  → 18048e5f… (verification fixes: license=SPDX string, explicit namespace data packages).
  Still the same single spec-mandated meta-file exception; re-baselined, not silent.
- requirements-dev.txt unchanged since v0.15 packaging (3d2f3e46…).
- rasm_cim/cli.py, rasm_cim/__main__.py, tests_phase_c/, rasm_cim_rust/tests/conftest.py,
  tests_v15 regression test — NEW files (additive), not part of the v0.10 baseline set.
- rasm_cim_v15/learned/current.json — pointer stayed on lv_000 (md5 5a2c518a49f2559a84ad1aadf63c94da);
  an accidental in-repo train during a flawed PYTHONPATH experiment created lv_001.json,
  which was moved out of the repo (preserved at /tmp/verify_smoke/lv_001_accidental_repo_train.json),
  never committed. Rollback re-verified in an isolated package copy (see stage_e_rollback_v2.log).
- Verification artifacts live in results_verify/ (stage logs A–J).
