# Copilot Instructions

## Commands

Use the repository root as the working directory. `pytest.ini` sets `pythonpath = .`, so imports assume repo-root execution.

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy pytest
```

```bash
pytest tests/ -v
pytest tests/test_metrics.py -v
pytest tests/test_metrics.py::test_count_metastable_states_n3_two_maxima_baseline -v
```

## High-level architecture

- `shared/` is the theory-backed math layer. It holds the reusable Appendix A primitives: coarse-graining and inverse reconstruction, effective potentials, coupling tensors, brake variation, and the four Thread 7 measurement functions. Most new mathematical work should start here, and it should stay parameterized by polynomial order `n`.
- `models/dim_1plus1/` builds the executable 1+1 dynamics on top of `shared/`. `cfe.py` implements the fixed-metric coherence equation and Laplace-Beltrami stencil; `mfe.py` implements the metric equation plus the coupled solver. The public entry points are re-exported from `models.dim_1plus1`, especially `run_simulation()`.
- The coupled solver in `models/dim_1plus1/mfe.py` evolves `concat(C, log g)` rather than `(C, g)`. That representation is deliberate: it keeps the metric positive, the stored state aligned with the metric used by the CFE/MFE RHS functions, and supports the default metric floor/ceiling events.
- `tests/` are the executable spec for the formalism. They do more than smoke-test numerics: they encode branch behavior in reconstruction, the n=3 analytical brake match, the expected n>3 mismatch trend, and the four Thread 7 measurements.
- `experiments/polynomial-sweep/` is the intended research driver layer for Thread 7, but the current executable substance is still concentrated in `shared/` and `models/dim_1plus1/`. Treat the experiment scripts there as scaffolding unless they are expanded.

## Key conventions

- Treat this repository as **scientific instrumentation**, not a generic PDE or ML codebase. Functions are expected to correspond to named objects from Semantic Relativity Appendix A, and naming should stay aligned with the theory.
- Use the Appendix A symbol mapping used by this repo, not older toy-model names: `gamma` replaces old `alpha`, `gaussian_kernel` / `G_sigma` replaces `K_sigma`, and `zeta` replaces old `beta`.
- Keep math in `shared/` general across polynomial orders `n = 2..6`. Avoid introducing cubic-only implementations unless the code is explicitly representing the special n=3 closed-form case already recognized by the theory and tests.
- Preserve explicit parameter passing. The core parameters (`sigma`, `gamma`, `mu_sq`, `alpha_phi`, `lambda_B`, `eta_g`, `xi_g`, plus grid terms like `dx`) are passed through function arguments or `params` mappings; there are no hidden globals.
- `shared.reconstruction.reconstruct()` has branch semantics that matter: for even `n`, it returns the principal monotone branch `psi > psi_c`; values below the branch minimum warn and return `NaN`. Do not “simplify” this into an unrestricted inverse.
- The CFE uses `+V_eff'(C)` as its reaction term, so measurement code interprets stable states as **maxima** of the effective landscape, not minima.
- In `models/dim_1plus1.cfe.cfe_rhs()`, `lambda_B == 0` intentionally skips reconstruction and brake evaluation to avoid unnecessary per-grid-point root finding. Preserve that fast path.
- Spatial derivatives use periodic centered-difference stencils (`np.roll`) consistently across the CFE and MFE code. New discretized operators should match that periodic-grid convention unless there is an explicit reason to change the scheme everywhere.
- `models/2plus1/` and `models/3plus1/` are future scaffolding. Do not populate them unless the task is explicitly about extending beyond the current 1+1 implementation.
