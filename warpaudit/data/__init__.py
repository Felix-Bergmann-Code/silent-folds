"""Images and metadata, grouping, and the development reserve.

The restricted annotation loader lives in :mod:`warpaudit.data.annotations`
and is deliberately not re-exported here, so that ``from warpaudit.data import
*`` inside a feature module cannot reach ground truth.
"""

from .development import DevelopmentManifest, select_development_groups
from .grouping import GroupAssignment, GroupingReport, assign_groups, grouping_claim
from .loaders import DatasetUnavailable, PairListing, get_loader
from .manifest import (
    CheckpointExposure,
    DatasetProvenance,
    PairRecord,
    ProvenanceManifest,
)

__all__ = [
    "CheckpointExposure",
    "DatasetProvenance",
    "DatasetUnavailable",
    "DevelopmentManifest",
    "GroupAssignment",
    "GroupingReport",
    "PairListing",
    "PairRecord",
    "ProvenanceManifest",
    "assign_groups",
    "get_loader",
    "grouping_claim",
    "select_development_groups",
]
