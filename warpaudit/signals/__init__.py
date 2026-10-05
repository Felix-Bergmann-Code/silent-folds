"""Ground-truth-free registration quality signals.

Importing this package registers every implemented family.  Registration is
kept explicit here: callers should not have to know which implementation
module contains a family, and adding a module must not silently make a
configured family unavailable.
"""

from . import family_a_appearance as _family_a_appearance  # noqa: F401
from . import family_b_correspondence as _family_b_correspondence  # noqa: F401
from . import family_c_cycle as _family_c_cycle  # noqa: F401
from . import family_d_plausibility as _family_d_plausibility  # noqa: F401
from . import family_e1_stability as _family_e1_stability  # noqa: F401
from . import family_e2_perturbation as _family_e2_perturbation  # noqa: F401
from . import family_f_disagreement as _family_f_disagreement  # noqa: F401
from . import family_g_structure as _family_g_structure  # noqa: F401
from .registry import available_families, compute_families, get_signal

__all__ = ["available_families", "compute_families", "get_signal"]
