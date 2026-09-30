# Silent Folds

Code and results for

> F. Bergmann, "Silent Folds: Projective Poles as a Training-Free Failure Flag
> in Homography-Based Retinal Registration," submitted to IEEE ISBI 2027.

Every homography has a pole line on which its projective denominator vanishes.
Where that line crosses the fundus, the warp folds while the matrix still looks
valid. This repository contains the exact support tests (rectangle, circular
field of view, convex mask), the scale-invariant pole clearance ρ, the
support-constrained RANSAC, and the full audit of XFeat, SuperPoint/LightGlue,
SIFT and SuperRetina registrations on FIRE and COph100 that the paper reports.

The pole geometry itself is in
[`warpaudit/geometry/projective.py`](warpaudit/geometry/projective.py) and
depends only on NumPy.

## Repository layout

| Path | Contents |
|---|---|
| `warpaudit/` | Evaluation package: data manifests, registration adapters, shared seeded RANSAC/DLT, pole geometry, QC features, grouped detector fitting, metrics |
| `scripts/` | Experiment and paper-building scripts (see below); `scripts/matchers/` and `scripts/superretina/` are the isolated registration workers |
| `configs/` | Frozen study configuration (`full_study.yaml`) and detector design files |
| `manifests/` | Pair manifest, patient/component groups and data provenance (paths and hashes only, no images) |
| `reports/` | Committed per-case and aggregate results behind every number in the paper |
| `benchmark_pole_outputs*/` | Pole prevalence on non-retinal benchmarks (supplementary; not in the 4-page paper) |
| `paper/isbi_2027/` | LaTeX source of the paper; `numbers.tex` is generated |
| `tests/` | Unit and integration tests (`pytest`) |

## Installation

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -c requirements-eval.lock -e '.[dev,report]'
python -m pytest
```

The matchers run in separate environments so that their Torch/CUDA stacks do
not leak into the evaluation environment; `scripts/setup_matcher_envs.sh`
creates them from `requirements-xfeat.lock` and `requirements-sp-lightglue.lock`.

## Reproducing the paper

### From the committed results (minutes, no images or GPU needed)

```bash
python scripts/isbi_review_analyses.py        # AUROC intervals, paired differences,
                                              # threshold sensitivity, Brier/NLL
python scripts/build_isbi_numbers.py          # writes paper/isbi_2027/numbers.tex
python scripts/build_isbi_near_pole_figure.py # synthetic near-pole sweep
python scripts/build_isbi_clearance_figure.py # Fig. 2
python scripts/isbi_detector_fold_auroc.py    # within-fold detector vs. inlier ratio
cd paper/isbi_2027 && latexmk -pdf main.tex
```

`numbers.tex` and `reports/isbi_review_analyses/` regenerate byte-identically
from the committed inputs.

| Paper item | Script | Committed output |
|---|---|---|
| Table 1: crossings, failures, ρ ≤ 1 | `scripts/isbi_submission_experiments.py`, `scripts/isbi_superretina_audit.py` | `reports/pole_guard_ablation_latest/pole_clearance.csv`, `reports/isbi_submission_latest/` |
| Table 1: AUROCs and 95% group-bootstrap intervals | `scripts/isbi_review_analyses.py` | `reports/isbi_review_analyses/auroc_ci.csv` |
| Table 1: constrained RANSAC | `scripts/isbi_submission_experiments.py` | `reports/isbi_submission_latest/constrained_ransac/` |
| Table 1: detector AUROC; Table 2 | `scripts/pole_guard_ablation.py` | `reports/pole_guard_ablation_latest/` |
| Table 2: Brier and NLL | `scripts/isbi_review_analyses.py` | `reports/isbi_review_analyses/calibration_quality.csv` |
| Paired AUROC differences, threshold sensitivity | `scripts/isbi_review_analyses.py` | `reports/isbi_review_analyses/` |
| SuperRetina FIRE reproduction | `scripts/isbi_superretina_audit.py` | `reports/isbi_submission_latest/superretina/fire_official_reproduction.json` |
| FIRE standard success-curve protocol | `scripts/fire_standard_protocol.py` | `reports/fire_standard_protocol_latest/` |
| Fig. 1 (needs COph100/RIDIRP images) | `scripts/build_qualitative_pole_figure.py` | `reports/feedback_revision_latest/fov_pole_cases.json` |
| Fig. 2 | `scripts/build_isbi_clearance_figure.py` | `paper/isbi_2027/assets/clearance_qc.pdf` |

### From scratch (datasets and a CUDA GPU)

1. Download and verify the datasets (see [`DATA_LICENCES.md`](DATA_LICENCES.md)):

   ```bash
   python -m warpaudit prepare-data --config configs/pilot.yaml --dataset COph100 --download
   python -m warpaudit prepare-data --config configs/pilot.yaml --dataset FIRE --download \
     --acknowledge-fire-terms-unresolved
   python -m warpaudit audit-data --config configs/pilot.yaml
   ```

2. Run the registrations and features of the frozen study
   (`scripts/run_full_study.ps1` on Windows, `scripts/run_study.sh` elsewhere).
   This caches every correspondence set and homography.
3. Run the pole-guard ablation and the ISBI experiments:
   `scripts/pole_guard_ablation.py`, then `scripts/run_isbi_submission.ps1`
   (clearance QC, constrained RANSAC, and SuperRetina; the SuperRetina stage
   needs the released `SuperRetina.pth`, see the script header).
4. Continue with the steps in the previous section.

All RANSAC runs are seeded, so the generic-pipeline homographies replay
bit for bit from the cached correspondences.

## Data

No images, annotations or pretrained weights are redistributed. FIRE, COph100
and RIDIRP are downloaded from their official sources and checked against
pinned SHA-256 hashes; see [`DATA_LICENCES.md`](DATA_LICENCES.md). The one exception is the paper's
Fig. 1, which shows two CC0 RIDIRP images.

## Citation

See [`CITATION.cff`](CITATION.cff). Please also cite FIRE, COph100 and RIDIRP
when you use their data.

## License

Code: [MIT](LICENSE). Datasets and pretrained weights keep their own terms.
