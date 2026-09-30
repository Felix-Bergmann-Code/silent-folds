"""Registration fitting and resumable adapter execution."""

from .fitting import FitResult, FittingPolicy, fit_homography
from .runner import (
    RegistrationJob,
    RegistrationRunner,
    deserialise_transform,
    serialise_transform,
)
from .subprocess_adapter import AdapterProtocolError, SubprocessMatcherRegistrar

__all__ = [
    "AdapterProtocolError",
    "FitResult",
    "FittingPolicy",
    "RegistrationJob",
    "RegistrationRunner",
    "SubprocessMatcherRegistrar",
    "deserialise_transform",
    "fit_homography",
    "serialise_transform",
]
