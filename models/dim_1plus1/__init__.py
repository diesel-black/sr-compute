"""1+1 SR model: coupled CFE and MFE time evolution (Appendix A.1.6–A.1.8)."""

from .cfe import cfe_rhs, integrate_cfe, laplace_beltrami_1d
from .mfe import coupled_rhs, initial_conditions, integrate_coupled, mfe_rhs, run_simulation

__all__ = [
    "cfe_rhs",
    "integrate_cfe",
    "laplace_beltrami_1d",
    "coupled_rhs",
    "initial_conditions",
    "integrate_coupled",
    "mfe_rhs",
    "run_simulation",
]
