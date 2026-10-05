"""One unattended pass over every executable stage of the study.

The repository deliberately exposes execution through the ``warpaudit`` CLI so
that job identifiers, provenance hashes, atomic shards, and the status ledger
have exactly one implementation. This module therefore does not reimplement any
stage: it declares the dependency order between existing commands, so a single
overnight invocation runs them in a sequence whose prerequisites are explicit
and whose failures name the stages they invalidate.

Two boundaries are structural, not incidental:

* label access stops at the development split. Confirmatory registrations and
  features are ground-truth-free and may be computed ahead of the freeze;
  confirmatory *outcomes* may not, so no stage here reads them.
* E1 is excluded from the confirmatory feature sweep because the development
  gate froze it as degenerate (§6.4). Recomputing it over the reserve would
  cost hours and could not enter the common block.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from .config import Config

__all__ = ["Stage", "StagePlan", "build_plan"]


@dataclass(frozen=True)
class Stage:
    """One named unit of the run, executed as a sequence of CLI invocations."""

    id: str
    title: str
    #: Argument vectors passed to the ``warpaudit`` CLI, in order.
    commands: tuple[tuple[str, ...], ...]
    #: Stage ids that must have succeeded first.
    requires: tuple[str, ...] = ()
    #: What the stage produces, and why later stages need it.
    purpose: str = ""
    #: Stages whose cost is dominated by matcher execution, for the plan summary.
    heavy: bool = False
    #: Optional stages are reported but never block the stages that follow.
    optional: bool = False


@dataclass(frozen=True)
class StagePlan:
    stages: tuple[Stage, ...] = field(default_factory=tuple)

    def __iter__(self):
        return iter(self.stages)

    def __len__(self) -> int:
        return len(self.stages)

    def ids(self) -> tuple[str, ...]:
        return tuple(stage.id for stage in self.stages)

    def get(self, stage_id: str) -> Stage:
        for stage in self.stages:
            if stage.id == stage_id:
                return stage
        raise KeyError(stage_id)

    def select(
        self,
        *,
        only: tuple[str, ...] = (),
        skip: tuple[str, ...] = (),
        start: str = "",
    ) -> StagePlan:
        """Narrow the plan while keeping declared order.

        Prerequisites are *not* silently pulled in: a caller that skips a stage
        is stating that its outputs already exist, and the run records which
        stage each later one depended on so that claim stays visible.
        """
        unknown = (set(only) | set(skip) | ({start} if start else set())) - set(self.ids())
        if unknown:
            raise ValueError(f"unknown stage id(s): {sorted(unknown)}")
        stages = list(self.stages)
        if start:
            stages = stages[self.ids().index(start) :]
        if only:
            stages = [s for s in stages if s.id in set(only)]
        if skip:
            stages = [s for s in stages if s.id not in set(skip)]
        return StagePlan(tuple(stages))


def build_plan(
    cfg: Config,
    *,
    config_path: str,
    download: bool = False,
    acknowledge_fire_terms: bool = False,
    include_confirmatory: bool = True,
    development_families: tuple[str, ...] = (),
    confirmatory_families: tuple[str, ...] = (),
    e2_limit: int = 1,
    e2_sample_cap: int = 0,
    signed_off_by: str = "",
    review_note: str = "",
    acknowledge_infeasible: bool = False,
    refit_bootstrap: int = 0,
    feature_workers: int = 1,
    feature_worker_threads: int = 0,
) -> StagePlan:
    """Declare the full executable study in dependency order."""
    common = [p.id for p in cfg.pipelines if p.in_common_block]
    if feature_workers < 1:
        raise ValueError("feature_workers must be positive")
    if feature_worker_threads < 0:
        raise ValueError("feature_worker_threads must be nonnegative")
    if not common:
        raise ValueError("no common-block pipeline is configured")
    base = ("--config", config_path)
    # Development extracts every configured family, because the degeneracy gate
    # and the diagnostics are decided there. The confirmatory sweep extracts the
    # cheap families only: E1 is the run's largest avoidable cost and the gate
    # can remove it, so it is requested for the reserve only when the frozen
    # families actually include it.
    development_families = tuple(development_families or cfg.features.families)
    if not confirmatory_families:
        confirmatory_families = tuple(
            f for f in development_families if f not in {"E1", "E2"}
        )

    def prepare_commands() -> tuple[tuple[str, ...], ...]:
        out: list[tuple[str, ...]] = []
        for ds in cfg.datasets:
            # A resumed run must not fail on data it already prepared, so the
            # plan always reuses an existing destination; the archives are still
            # verified and audit-data still re-checksums every extracted file.
            argv = ["prepare-data", *base, "--dataset", ds.id, "--reuse-existing"]
            if download:
                argv.append("--download")
            # The flag is a licence acknowledgement, not a convenience switch;
            # FIRE extraction refuses without it whether or not it downloads.
            if ds.id == "FIRE" and acknowledge_fire_terms:
                argv.append("--acknowledge-fire-terms-unresolved")
            out.append(tuple(argv))
        return tuple(out)

    stages: list[Stage] = [
        Stage(
            id="environment",
            title="Record the evaluation environment and code identity",
            commands=(
                ("environment", *base, "--output",
                 (Path(cfg.paths.reports) / "environment.json").as_posix()),
                ("validate-config", *base),
            ),
            purpose="Pins the interpreter, library versions, git commit, and the "
            "registration/feature code identities this run's caches are keyed by.",
        ),
        Stage(
            id="probe-adapters",
            title="Probe each matcher environment on a synthetic pair",
            commands=tuple(
                ("probe-adapter", *base, "--pipeline", pipeline_id, "--output",
                 (Path(cfg.paths.reports) / f"probe_{pipeline_id}.json").as_posix())
                for pipeline_id in common
            ),
            requires=("environment",),
            purpose="Fails in seconds if an isolated environment, checkpoint, or "
            "coordinate convention is wrong, instead of after hours of sweeping.",
        ),
        Stage(
            id="prepare-data",
            title="Reconstruct datasets from checksum-pinned official archives",
            commands=prepare_commands(),
            requires=("environment",),
            purpose="Verifies archive checksums and extracts images and annotations. "
            "Nothing downstream may run against unverified bytes.",
        ),
        Stage(
            id="audit-data",
            title="Audit datasets and write identity manifests",
            commands=(("audit-data", *base),),
            requires=("prepare-data",),
            purpose="Writes pairs, groups, provenance, and the development split. "
            "Every later stage selects work from this manifest.",
        ),
        Stage(
            id="register-development",
            title="Register the development split in both directions",
            commands=tuple(
                ("register", *base, "--split", "development", "--pipeline", pipeline_id,
                 "--direction", "both")
                for pipeline_id in common
            ),
            requires=("audit-data", "probe-adapters"),
            purpose="Forward and independently estimated reverse transforms. The "
            "reverse rows are the cycle signal's only honest input.",
            heavy=True,
        ),
        Stage(
            id="labels-development",
            title="Score development registrations in the restricted evaluator",
            commands=tuple(
                ("labels", *base, "--split", "development", "--pipeline", pipeline_id,
                 "--direction", "canonical")
                for pipeline_id in common
            ),
            requires=("register-development",),
            purpose="Development outcomes only. Confirmatory labels stay unread: "
            "no command in this plan can reach them.",
        ),
        Stage(
            id="features-development",
            title="Extract development signal families",
            commands=(
                ("features", *base, "--split", "development", "--families",
                 *development_families, "--direction", "canonical"),
            ),
            requires=("register-development",),
            purpose="Includes E1 because the degeneracy gate must be recomputed "
            "from current code before it can be relied on again.",
            heavy=True,
        ),
        Stage(
            id="e2",
            title="Run coordinate-corrected input-perturbation reruns",
            commands=tuple(
                (
                    "e2",
                    *base,
                    "--pipeline",
                    pipeline_id,
                    "--split",
                    "development",
                    *(("--limit", str(e2_limit)) if e2_limit else ()),
                    *(("--sample-cap", str(e2_sample_cap)) if e2_sample_cap else ()),
                )
                for pipeline_id in common
            ),
            requires=("register-development",),
            purpose="Bounded development smoke test: full perturbed reruns, so its "
            "cost scales with the rerun count and not with the split size.",
            heavy=True,
        ),
        Stage(
            id="diagnose-development",
            title="Freeze the cycle and E1 informativeness decisions",
            commands=(("diagnose-development", *base),),
            requires=("features-development", "labels-development"),
            purpose="Writes the feature gate that decides which families the "
            "common block is allowed to use.",
        ),
    ]

    if include_confirmatory:
        stages += [
            Stage(
                id="register-confirmatory",
                title="Register the confirmatory reserve in both directions",
                commands=tuple(
                    ("register", *base, "--split", "confirmatory", "--pipeline", pipeline_id,
                     "--direction", "both")
                    for pipeline_id in common
                ),
                requires=("audit-data", "probe-adapters"),
                purpose="Label-free computation over the reserve. Permitted before "
                "the freeze precisely because no outcome is read here.",
                heavy=True,
            ),
            Stage(
                id="features-confirmatory",
                title="Extract frozen signal families over the reserve",
                commands=(
                    ("features", *base, "--split", "confirmatory", "--families",
                     *confirmatory_families, "--families-from-gate",
                     "--direction", "canonical"),
                ),
                requires=("register-confirmatory", "diagnose-development"),
                purpose="Frozen families only. E1 is excluded by the development "
                "gate, which removes the run's largest avoidable cost.",
                heavy=True,
            ),
        ]

    stages += [
        Stage(
            id="plan-study",
            title="Screen both transfer directions for information feasibility",
            commands=(("plan-study", *base),),
            requires=("labels-development",),
            purpose="Decides whether any direction can carry the joint claim, and "
            "names the blocking criteria when none can.",
        ),
    ]

    # G1 and everything after it read confirmatory outcomes, so they exist in
    # the plan only when a reviewer has signed the freeze. Without that the run
    # stops at the screen rather than quietly proceeding past the gate.
    if include_confirmatory and signed_off_by and review_note:
        stages += [
            Stage(
                id="freeze",
                title="Record the reviewed G1 freeze",
                commands=(
                    (
                        "freeze",
                        *base,
                        "--signed-off-by",
                        signed_off_by,
                        "--review-note",
                        review_note,
                        *(("--acknowledge-infeasible",) if acknowledge_infeasible else ()),
                    ),
                ),
                requires=("plan-study", "diagnose-development"),
                purpose="Fixes direction, folds, budget, families, learner, policy, "
                "and margins, and gates confirmatory outcome access on them.",
            ),
            Stage(
                id="labels-confirmatory",
                title="Score the confirmatory reserve",
                commands=tuple(
                    ("labels", *base, "--split", "confirmatory", "--pipeline", pipeline_id,
                     "--direction", "canonical")
                    for pipeline_id in common
                ),
                requires=("freeze", "register-confirmatory"),
                purpose="The first stage permitted to read reserve outcomes, and only "
                "because the freeze that gates it already exists.",
            ),
            Stage(
                id="evaluate",
                title="Fit, freeze, and test the joint claim",
                commands=(
                    (
                        "evaluate",
                        *base,
                        *(("--refit-bootstrap", str(refit_bootstrap)) if refit_bootstrap else ()),
                    ),
                ),
                requires=("labels-confirmatory", "features-confirmatory"),
                purpose="Runs the folded protocol and the complete refit bootstrap "
                "behind the three one-sided component bounds.",
                heavy=True,
            ),
            Stage(
                id="failure-gallery",
                title="Render the deterministic failure gallery",
                commands=(("failure-gallery", *base),),
                requires=("evaluate",),
                purpose="Selected after the numeric results, under a disclosed rule; "
                "panels stay in the ignored local review directory.",
                optional=True,
            ),
        ]

    stages += [
        Stage(
            id="report",
            title="Build manuscript tables and figures",
            commands=(("report", *base),),
            requires=("audit-data",),
            purpose="Derives every table and figure from the caches, so a reported "
            "number can be traced to the row that produced it.",
        ),
        Stage(
            id="estimate-cost",
            title="Summarise measured runtimes",
            commands=(("estimate-cost", *base),),
            requires=("register-development",),
            purpose="Reports the run's own measured cost basis rather than an "
            "assumed one.",
        ),
    ]
    if feature_workers > 1 or feature_worker_threads:
        stages = [
            replace(stage, commands=tuple(
                (*command, "--workers", str(feature_workers), "--flush-every", "1")
                + (("--worker-threads", str(feature_worker_threads)) if feature_worker_threads else ())
                if command[0] == "features" else command
                for command in stage.commands
            ))
            for stage in stages
        ]
    return StagePlan(tuple(stages))
