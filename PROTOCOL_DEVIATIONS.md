# Protocol deviations

No frozen protocol exists yet. Engineering changes before G1 belong in version
history and do not count as deviations from a preregistration.

After G1, append (never rewrite) entries with date, affected artifact/hash,
reason, whether target outcomes had been inspected, and which results must be
labelled exploratory.

## 2026-09-08T14:46:46+00:00 -- matcher provenance re-pinned for this machine

Host: Windows-10-10.0.26200-SP0 (AMD64).

- `xfeat_h.python_version`: `3.11.5` -> `3.11.11`
- `xfeat_h.torch_version`: `2.3.1` -> `2.3.1+cu121`
- `sp_lg_h.python_version`: `3.11.5` -> `3.11.11`
- `sp_lg_h.torch_version`: `2.3.1` -> `2.3.1+cu121`

Upstream commits and checkpoint hashes are unchanged; only interpreter and library builds differ. The configuration hash moves with these pins, so this machine computes its own caches instead of reusing rows produced under a different stack.

## 2026-09-09 — pre-G1 failure recovery and computational rerun

The interrupted run failed while serializing an undefined diagnostic for a
legitimate `no_matches` registration. Diagnostics now encode nonfinite
measurements as JSON null while retaining the explicit status and reason.
Technical failures now receive the existing fixed three-attempt policy in the
CLI; unresolved jobs block downstream stages. Feature provenance now includes
registration code so repaired inputs cannot reuse stale derived features.
LightGBM is pinned to locally validated 4.6.0 after 4.7.0 crashed in the grouped
secondary-learner test on this Windows machine.

No G1 freeze or confirmatory label cache existed at recovery. Confirmatory
ground truth was not read during diagnosis. Preserve the interrupted artifacts
and rebuild registrations, development labels/features/gates, and subsequent
stages from a fresh cache, reusing verified data. Configuration, splits, seeds,
weights, fitting settings, endpoints and bootstrap counts are unchanged. These
are pre-freeze engineering repairs, not amendments to a frozen protocol.
Evidence and validation are recorded in `reports/STUDY_RECOVERY.md`.

## 2026-09-09 — parallel feature scheduling before G1

At the user's request, resume feature extraction with four independent case
workers and one cache writer. Seeds, scientific algorithms, numerical thread
settings, splits and bootstrap counts remain unchanged. Four saved real cases
reproduced all 312 feature values exactly under the new scheduler; the full
suite passed (96 passed, one platform skip). Preserve 556 registrations and
25 saved feature cases; recompute the serial process's 13 unflushed cases.
No confirmatory labels were accessed. This is an execution-only change, with
benchmark evidence and restart records in `reports/STUDY_RECOVERY.md`.

## 2026-09-09 ? fresh isolated 32-worker run

User requested deletion of prior runs and a full fresh computation. Superseded
caches, manifests, logs and result artifacts were deleted after validation.
Current run is fresh32-20260909T090621Z, with isolated output paths and a new
configuration hash. Scientific settings remain unchanged. Features use 32 case
workers and one native library thread each; 2,496 real feature values matched
serial results exactly. Full validation: 98 passed, one platform skip. The new
run stores source/configuration/package snapshots and validation evidence in
its own study_outputs directory. Earlier audit entries remain historical; their
referenced old run files no longer exist by user request.

## 2026-09-09 — outer-fold count reduced from 3 to 2 before G1

The 2026-09-09 run stopped at the M2 feasibility gate: neither transfer
direction met the class-support screening minima, so no primary direction could
be frozen. `reports/FRESH_STUDY.json` records `confirmatory_evaluation_started:
false`; no confirmatory outcome was read, so this is a pre-freeze design change,
not an amendment to a frozen protocol.

`splits.min_outer_folds`: `3` -> `2`, in `configs/pilot.yaml` and
`configs/fresh32-20260909T090621Z.yaml`. `n_outer_folds` stays 5, so
`choose_fold_count` now falls back to 2 for the 44-group COph100 confirmatory
target instead of 3.

Reason. `_matched_budget` caps the per-fold budget by the FIRE source capacity
(`33 - 10 = 23`), not by the target pool, so 2 folds keep the same 11 training
and 11 calibration groups per side and the same 11 independent source test
groups; only the smallest target test fold changes, from 14 groups to 22. Every
confirmatory target group is still tested exactly once, so the evidence base is
unchanged. `_aggregate_folds` leaves an estimand undefined unless every fold
contributed, so a single class-short fold discards the primary endpoint
entirely; the probability that every fold of both pipelines is defined rises
from about 0.76 to about 0.97. This is the third admissible response the
feasibility screen itself names.

This does not change the joint claim's power, which the M2 scenarios put at
0.001-0.009 under every assumption considered. That limitation is unaffected by
the fold count and is not addressed here.

Target outcomes had not been inspected. No results become exploratory. The
scientific settings -- data, seeds, weights, fitting, endpoints, bootstrap
counts, labels, features -- are unchanged.

Cache impact. This edit moves the configuration hash from `060f3cc638751e20`
to `e50494c3aecd5c6d`. Registration rows are gated on `config_hash` by
`_cache_contract_matches`, and `feature_id` hashes the configuration outright,
so both caches would be discarded even though neither can change. Labels carry
no `config_hash` and are unaffected. The new `warpaudit rekey-cache` command
carries registration and feature rows onto the current contract, and refuses to
do so unless the two configurations differ exclusively in fields that cannot
change a cached value.

## 2026-09-09: reviewed two-fold pilot freeze and outcome access

After the regenerated development-only M2 screen recommended `FIRE->COph100`
with two folds and no blocking criteria, the user explicitly selected the
low-powered pilot route. G1 freeze `fb7ea814e66d2164` records Felix Bergmann's
sign-off and the pre-G1 three-to-two-fold amendment before the authorized resume
at `labels-confirmatory`. No infeasibility override was used. The cache metadata
audit immediately preceding this continuation found no confirmatory label IDs
and no prior G1 freeze; details and preservation limitations are recorded in
`reports/STUDY_RECOVERY.md`.

The regenerated recommended-direction joint-pass design scenarios span
0.00065-0.01505 (approximately 0.065%-1.505%), superseding the earlier range
above. The study proceeds as a severely underpowered pilot: report estimates
and intervals, including inconclusive outcomes. The two-fold amendment improves
class support without adding independent subjects and is a substantive protocol
change. It does not justify a well-powered confirmatory headline claim.


## 2026-09-09: public-data replacement before external outcomes

At the user's request, replace inaccessible ANHIR and AN-200 with the public
MultiRegEval histology and cytology releases, Zenodo 5557568 v1.2.0 (CC BY 4.0).
Archive MD5 checksums matched publisher records; local SHA-256 checksums and
index hashes are in manifests/public_external_acquisition.json. MEMO remains
local but is omitted from this active configuration pending grouping/licence review.

This is a substantive scope change: the new external tests measure controlled
rigid transformations on biomedical images. Repeated cores, frames, crops and
severities do not establish independent clinical evidence. Histology is pooled
as one cohort because donor mapping is absent; cytology uses three cell-line
groups. Switch explicitly to descriptive mode; retain the confirmatory minima
for any future confirmatory design. Suppress sparse-group bootstrap intervals,
and report no confirmatory primary claim. The reviewed freeze and development
model/provenance checks remain required. Windows interpreter paths were corrected;
all reported runtime versions, upstream commits and weights matched existing pins.


## 2026-09-10 ? parallel execution and E2 coordinate-map repair

At the user's request, replace serial per-case execution with bounded spawned
workers for registration, E2 preparation/reruns, label scoring, feature
extraction, and per-pipeline development fitting/external evaluation. Audit
file hashing uses bounded I/O workers. Cache and ledger publication remain
single-writer and ordered; seeds, dataset partitions, clinical/grouping rules,
replicate counts, calibration splits, eligibility gates, and estimands are
unchanged. Pipeline sweeps share a worker budget; dependency barriers remain.

Inspection found an existing E2 bug: maps were keyed by perturbation condition
alone, so the last prepared pair overwrote earlier pairs' coordinate corrections
for the same draw index. Each E2 job now carries its own generated input maps;
the correction adapter keys them by both pair and condition. This intentionally
corrects earlier E2 values. Registration, E2 and dependent feature provenance
invalidate affected caches. The previous interrupted run must be preserved
separately, and the replacement study starts with an empty cache. No external
outcomes were used to select this repair. This does not rerun the completed
pilot as a new confirmatory experiment; the active study remains descriptive.

Serial-versus-parallel checks compare scientific values, excluding timestamps
and resource timings. E2 diagnostics are additionally checked against each
pair's independently reconstructed seeded translations. Parallel runtime
measurements are recorded as contended throughput measurements and must not
be presented as isolated per-case latency.

## 2026-09-13 - original-frame label validity repair

The conference development audit identified 20 SIFT/FIRE cases with undefined
bounded loss. Each has ten finite annotated landmarks, a working-frame fit that
passes the existing validity gate, and finite mapped landmarks. Collapsing the
original-to-working/fit/working-to-original composition changed the raw matrix
singular-value ratio across the 1e-10 cutoff. Label scoring then treated these
accepted working fits as undefined ground truth, with operational_failure=False.
Direct forward evaluation gives mean landmark errors of about 986-1709 original
pixels for these cases; they must be retained as geometric failures.

Original-frame conversion now retains the composition and validates its
constituents, including the fit in its working frame. Non-finite mapped
landmarks remain rejected. Registration algorithms, fitted transforms, feature
values, split seeds and thresholds are unchanged. All canonical development
labels are recomputed, followed by the development review and information plan.
Earlier preflight results and any plans based on the old labels are superseded.

Geometry files participate in broad cache identities. The one-off
`scripts/repair_coordinate_labels.py` verifies the exact conversion-only AST
change and all pre-repair base/stage identities before migrating cache identity
columns, including the derived feature hashes and feature IDs. Every original
shard is backed up, and every non-identity column is
verified equal after writing. Local source snapshots, old/new identities,
original shards, per-case diagnosis and label changes are retained under
`conference_outputs/coordinate_repair/` (the initial per-case diagnosis is under
`conference_outputs/review_v2/coordinate_diagnosis.json`). This is an explicit
cache migration for unchanged registration/E2/feature computations, not a
claim that old labels remain valid. The script refuses existing study freezes
or extension locks. No external outcomes were used to select the repair, and
no scientific reviewer sign-off is implied by these software checks.

## 2026-09-13 — Separate exploratory development decision analysis

Added `scripts/development_decision.py` and `configs/development_decision.json`
after inspecting the corrected SIFT class counts. This is a development-informed
scope decision, not an amendment claiming untouched confirmation. It evaluates
all existing component arms in fixed nested group partitions over FIRE/COph100.
XFeat/SP-LightGlue homographies are primary; SIFT and TPS remain secondary.
Infeasible arm/folds are explicitly non-estimable; no label, seed, failure
threshold, cached measurement, or base protocol has been changed. The new runner
is separate from the prospective external extension and its freeze gates.

Conditional group bootstrap intervals hold held-out development predictions
fixed and exclude training/model-selection variability. They guide feasibility,
not confirmatory significance. Natural external-cohort access remains unverified.
See `reports/DEVELOPMENT_DECISION_RUN.md` for the complete executable contract.

## 2026-09-13 — Exploratory calibration/transport follow-up

The mixed development decision results motivated a separate publication audit.
`scripts/publication_audit.py` reuses fixed source models for threshold-only
adaptation and target probability recalibration at nested 5-group, 10-group and
full-calibration-fold budgets. Target subsets are selected without outcomes and
never include test groups. Primary target pipelines remain XFeat homography and
SuperPoint/LightGlue homography. Original results, base cache identities and
freeze gates are unchanged. The follow-up is explicitly exploratory and records
negative calibration slopes without test-directed corrections. Natural external
validation and any final frozen protocol remain pending. See
`reports/PUBLICATION_AUDIT_RUN.md` for the complete analysis contract.

## 2026-09-13 — Paper evidence protocol after inspecting calibration audit

The manuscript direction changed from broad stability improvement to an audit
separating ranking, probability calibration and acceptance thresholds. Added
`scripts/paper_study.py` and `configs/paper_study.json`: fixed repeated target
budgets, training-bound clipped/nonnegative-slope calibration sensitivity, margin
contribution tracing and complete/partial result accounting. These choices were
made after development outcome inspection and remain exploratory. No base cached
measurements, labels, seeds, features, or existing freeze records are amended.

The optional external path introduces a **separate descriptive cohort lock**, not
a substitute G1/G2 confirmatory pass. It requires actual curator evidence, natural
landmarks, grouping review and an attestation of no prior outcome inspection;
rejects already cached cohort outcomes; and fixes cohort/cache/design identity and
calibration/test groups before computing local external labels. This is a new
outcome-access contract, not a claim that the old confirmatory gates were satisfied.
External labels remain in its isolated output checkpoint. Natural dataset access,
scientific appropriateness and information adequacy remain unresolved until reviewed.
The operational minimum group counts are not a prospective power calculation.
See `specifications/PAPER_EVIDENCE_PROTOCOL.md` and `reports/PAPER_STUDY_RUN.md`.

## 2026-09-13 — Mechanism isolation and independent-cohort execution workflow

After reviewing the paper-study results, the user requested experiments addressing
generality and registration-specific mechanism. Added a fixed exploratory 2×2
source-map factorial (training-bound clipping × slope constraint), isotonic and
prevalence references, source-only curvature ablations, full cached-transform
geometry diagnostics, and retrospective leave-one-dataset-out transport. These
are explicitly development-informed. They do not add independent subjects or
amend the original measured features, labels, groups, or registration settings.

Added `paper_external_workflow.py` to complete the previously manual external
preparation path. It uses the original source cache and a separate target config
and cache. A truthful curator review and immutable preparation lock precede all
external E2 jobs. Those jobs use the existing ground-truth-free builder through
the new explicitly scoped descriptive entry point, instead of fabricating or
modifying G1/G2 records. A separate immutable evaluation lock fixes feature/cache
identity and calibration/test groups before local external labels are computed.
The original freeze functions and label caches are unchanged. Natural access,
annotation semantics and independent grouping are not supplied by code. See
`reports/EVIDENCE_CAMPAIGN_RUN.md` for the complete execution contract.
