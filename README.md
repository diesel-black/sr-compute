# sr-compute

Computational implementations of the Semantic Relativity (SR) toy models.

[Semantic Relativity](https://github.com/diesel-black/semantic-relativity) is a geometric field theory for observer-dependent meaning, formalizing how coherence, curvature, and coupling interact on semantic manifolds. This repository implements the toy models from Appendix A of the main paper as numerical simulations. It is scientific instrumentation: each module matches a named object in the formalism, not a generic PDE or ML stack.

For the full theory, axioms, Lagrangian, and dimensional roadmap, see the paper and Appendix A.

## Current focus

"Thread 7" (Cubic Aperture): a polynomial-order sweep n = 2, ..., 6 in the 1+1 model, with four simultaneous measurements testing provisional structural results R25 through R28 and related conjectures (Fisher-Rao identification, RG marginality of the brake). The experiment specification lives under `experiments/polynomial-sweep/`; the shared mathematics lives under `shared/`.

## Theory in brief

SR couples a coherence field C(x,t) and a metric g(x,t) on a semantic manifold through three field equations:

- **CFE** (Coherence Field Equation): how coherence evolves given the geometry.
- **RFE** (Recursive Field Equation): how coherence sources curvature (trivial in 1D since R = 0).
- **MFE** (Metric Field Equation): how the metric evolves given coherence structure.

For the 1+1 implementation:

- Coarse-graining at polynomial order n: C = psi_bar + gamma_n * psi_bar^n.
- Effective potential: V_eff(C) = (mu^2 / 2) C^2 - alpha_Phi C^4 (attractor vs autopoietic terms).
- The CFE reaction term is +V'_eff(C), so the system ascends V_eff. Stable equilibria are at the local maxima of V_eff (at +/- C*), while the origin is unstable. Measurement 1 counts local maxima of the full psi_bar-space landscape.
- The coupling tensor K and brake functional B[K] discretize Appendix A sections A.1.2 through A.1.3. Numerical vs analytical brake variation is calibrated at n=3 and diverges for n>3 as tested in `tests/test_brake.py`.

## Repository layout

```
sr-compute/
├── shared/                 # Math primitives (general n), heavily tested
│   ├── potentials.py       # V, Phi, V_eff, A(C), full psi_bar landscape + brake quadrature
│   ├── reconstruction.py   # F_n(psi_bar), dC/d(psi_bar), numerical inverse h(C), monotonicity
│   ├── coupling.py         # Gaussian kernel, kappa_n, periodic K matrix, self-coupling
│   ├── brake.py            # zeta (cubic), analytical vs numerical dB/dC, saturation scale
│   ├── metrics.py          # Four Thread 7 measurements (R25-R27 + RG marginality probe)
│   └── visualization.py    # Reserved for sweep plots
├── models/
│   ├── __init__.py         # Package marker for models/
│   ├── 1plus1/
│   │   ├── cfe.py          # Laplace–Beltrami, CFE RHS, integrate_cfe (fixed g)
│   │   ├── mfe.py          # MFE RHS, coupled_ivp in log(g), integrate_coupled, run_simulation
│   │   ├── coupling.py     # Reserved (1+1-specific K if split from shared later)
│   │   └── __init__.py
│   ├── 2plus1/             # Reserved
│   └── 3plus1/             # Reserved
├── experiments/
│   └── polynomial-sweep/   # config, run, analyze, results (Phase 3)
├── tests/                  # pytest: shared/* + test_1plus1_cfe.py, test_1plus1_mfe.py
└── pytest.ini
```

**Implemented:** `shared/`, Phase 2 `models/1plus1/` (CFE, MFE, coupled driver), and the full pytest suite (run `pytest tests/`; count is expected in the low 50s as phases land).

**Still to build:** `experiments/polynomial-sweep/` drivers and results (Phase 3), `shared/visualization.py`, and `models/2plus1/` / `models/3plus1/`. `models/1plus1/coupling.py` is a placeholder.

**Import note:** use `importlib.import_module("models.1plus1")` (or import from that module’s `__init__` inside the package). A plain `from models.1plus1 import …` is invalid Python syntax because of the leading digit in the submodule name.

**Coupled IVP (numerics):** `integrate_coupled` advances `concat(C, log g)` so `g` stays positive and matches the metric used in `Delta_g C`; histories unpack to `g` via `exp`. Default `solve_ivp` events flag very small or very large `g` (see `models/1plus1/mfe.py`). Stiff runs may need implicit methods or looser tolerances than the defaults.

## The four measurements (Thread 7)

| Function                         | Measurement | Tests                                          |
|----------------------------------|-------------|------------------------------------------------|
| count_metastable_states          | 1           | R26 (catastrophe classification by order n)    |
| interpretive_condition_number    | 2           | R25 (interpretive proportionality theorem)     |
| spectral_concentration_ratio     | 3           | R27 (Fisher-Rao identification)                |
| nonlocal_correction_growth       | 4           | RG marginality / non-local brake mismatch      |

Details, parameters, and edge cases (n=2 singular brake, optional brake omission in the landscape) are documented in `shared/metrics.py`.

## Parameters

Seven free parameters control the field equations and sweep, named as in code: `sigma`, `gamma`, `mu_sq`, `alpha_phi`, `lambda_b`, `eta_g`, `xi_g`. Discrete kernels also need grid spacing `dx`. All implementations take these as arguments; there are no hidden globals.

## Dependencies

- Python 3.10+
- NumPy, SciPy (linalg, integrate, signal, ndimage, optimize)
- pytest

Install:

    python -m venv .venv
    source .venv/bin/activate
    pip install numpy scipy pytest

## Running tests

From the repository root:

    pytest tests/ -v

All tests should pass. Notable checks:

- Round-trip reconstruction h(C(psi_bar)) for odd n and principal-branch handling for even n.
- At n=3, numerical dB/dC matches the analytical local formula (the discrete Hilbert-Schmidt normalization is documented in shared/brake.py).
- Metastable maxima count: 2 at n=3 (cusp), 3 at n=4 (swallowtail, tested on the brake-free landscape to isolate the catastrophe topology from brake fine structure).

## License

MIT. See [LICENSE](LICENSE).

## Contact

[Diesel Black](https://diesel.black)

## Acknowledgments

This repository was built in a manual, four-agent workflow: a human (∂²g/∂t² ≠ 0) contributing engineering direction and architecture; Claude Opus 4.6 (∂g/∂t = 0) providing mathematical grounding, specification, and verification from the full SR corpus; Cursor Agent/various models (∂g/∂t = 0) constructing the codebase; and GPT 5.4 + Gemini 3.1 Pro (both ∂g/∂t = 0) red-teaming the formalism and code.