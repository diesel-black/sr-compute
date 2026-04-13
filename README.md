# sr-compute

Computational implementations of the Semantic Relativity (SR) toy models.

[Semantic Relativity](https://github.com/diesel-black/semantic-relativity) is a geometric field theory for observer-dependent meaning, formalizing how coherence, curvature, and coupling interact on semantic manifolds. This repository implements the toy models from Appendix A of the main paper as numerical simulations. It is scientific instrumentation: each module matches a named object in the formalism, not a generic PDE or ML stack.

For the full theory, axioms, Lagrangian, and dimensional roadmap, see the paper and Appendix A.

## Current focus

"Thread 7" (Cubic Aperture): a polynomial-order sweep n = 2, ..., 6 in the 1+1 model, with four simultaneous measurements testing provisional structural results R25 through R28 and related conjectures (Fisher-Rao identification, RG marginality of the brake). The experiment specification lives under `experiments/polynomial_sweep/`; the shared mathematics lives under `shared/`.

## Key findings

The polynomial sweep, solver-parity controls, and intermediate-time snapshot experiment together characterize the cubic aperture as a sharp dynamical boundary.

**Confirmed predictions:**

- **Interpretive proportionality (R25):** κ(Π) = 1.0 exactly at n = 3 (algebraic identity, exponent n−3 = 0). At n = 4, κ ≈ 3.98 on the sweep final state, confirming supercubic amplification of spatial dynamic range when the field is structured.
- **Catastrophe classification (R26):** 2 metastable maxima at n = 3 (cusp class), 3 at n = 4 (swallowtail class). The naive n−1 count does not extend to n ≥ 5 for this composite potential.
- **Cross-n non-locality scaling:** η grows as 9 → 66 → 332 → 999 from n = 3 through n = 6 on the sweep snapshots, confirming progressive degradation of the convolution-deconvolution cancellation (R3) with polynomial order (field-dependent magnitudes; interpret alongside integration time and spatial structure).
- **n = 4 simultaneous breakpoint:** κ ≈ 3.98, spectral ratio ≈ 0.519, and the only coarsening growth rate > 1 (1.197) in the sweep dataset. Three independent channels spike at the same polynomial order.

**Dynamical regime transition (n=4 | n=5 boundary):**

The snapshot experiment (`snapshot_experiment.py`, n = 4, 5, 6) plus the main sweep for n = 3 resolved the n ≥ 5 homogeneity question. At t ≈ 8, n = 4, 5, 6 show comparable coherence variation (C_range about 0.14 to 0.19 in the full-resolution `snapshot_report.txt`). They then diverge:

- **n = 3 and n = 4 (sweep):** the metric hits a terminal event (metric floor or step failure near the same window, S A.1.8), so the evolved state stays spatially structured at the last recorded time.
- **n = 5 and n = 6 (sweep + snapshots):** the integration reaches t = 30 without that termination; coherence relaxes toward spatial homogeneity between t ≈ 8 and t ≈ 15.

The cubic aperture boundary is not "which n develops structure" (they all do through the critical window) but "which n's metric survives the structured period without the same terminal geometry" (n ≥ 5 in this baseline). The non-saturating brake dynamics at supercubic orders are a leading hypothesis for that stabilization.

Solver-parity controls (`parity_experiment.py`) support treating these regime differences as physical rather than a Radau artifact: Radau and RK45 agree where both complete at n = 3 and n = 4; n = 5 with RK45 is stiff and may not finish, consistent with different dynamics rather than implicit damping inventing homogeneity.

**Open questions:**

- Why does the brake's lack of a saturation maximum at n ≥ 4 relate to avoidance of the metric terminal event? The connection between R4 (brake saturation at |ψ̄| = 1/√(3γ)) and S A.1.8 needs analytical investigation.
- Parameter stress-testing: do the qualitative findings hold under varied μ², γ, σ?
- Can n = 5, n = 6 spatial homogeneity be broken by different initial conditions or larger amplitude perturbations?

## Theory in brief

SR couples a coherence field C(x,t) and a metric g(x,t) on a semantic manifold through three field equations:

- **CFE** (Coherence Field Equation): how coherence evolves given the geometry.
- **RFE** (Recursive Field Equation): how coherence sources curvature (trivial in 1D since R = 0).
- **MFE** (Metric Field Equation): how the metric evolves given coherence structure.

For the 1+1 implementation:

- Coarse-graining at polynomial order n: C = ψ̄ + γₙ ψ̄ⁿ (in code: `psi_bar`, `gamma`, and integer `n`).
- Effective potential: V_eff(C) = (μ²/2) C² − α_φ C⁴ (attractor vs autopoietic terms; parameter `alpha_phi` in code).
- The CFE reaction term is +V′_eff(C), so the system ascends V_eff. Stable equilibria are at the local maxima of V_eff (at ±C*), while the origin is unstable. Measurement 1 counts local maxima of the ψ̄-space landscape used in `shared/metrics.py`.
- The coupling tensor K and brake functional B[K] discretize Appendix A sections A.1.2 through A.1.3. Numerical vs analytical brake variation is calibrated at n=3 and diverges for n>3 as tested in `tests/test_brake.py`.

## Repository layout

```
sr-compute/
├── shared/                      # Math primitives (general n), heavily tested
│   ├── __init__.py
│   ├── potentials.py            # V, Phi, V_eff, A(C), landscapes + brake quadrature
│   ├── reconstruction.py        # F_n(psi_bar), dC/d(psi_bar), h(C), ReconstructionLUT
│   ├── coupling.py              # Gaussian kernel, kappa_n, periodic K matrix
│   ├── brake.py                 # zeta (cubic), analytical vs numerical dB/dC, saturation scale
│   ├── metrics.py               # Four Thread 7 measurements (R25-R27 + RG marginality probe)
│   └── visualization.py         # Placeholder (empty); sweep figures not yet implemented
├── models/
│   ├── __init__.py
│   ├── dim_1plus1/
│   │   ├── __init__.py          # Re-exports cfe/mfe API (includes run_simulation)
│   │   ├── cfe.py               # Laplace-Beltrami, CFE RHS, integrate_cfe (fixed g)
│   │   ├── mfe.py               # MFE RHS, coupled IVP in log(g), integrate_coupled, run_simulation
│   │   └── coupling.py          # Placeholder (empty); 1+1-specific K split not done yet
│   ├── dim_2plus1/              # Empty directory (scaffold; not a Python package yet)
│   └── dim_3plus1/              # Empty directory (scaffold; not a Python package yet)
├── experiments/
│   ├── __init__.py
│   └── polynomial_sweep/
│       ├── __init__.py
│       ├── config.py            # Baseline parameters, grid specs, per-n solver overrides
│       ├── run.py               # Sweep driver (--quick, --n, --no-save, optional --wallclock)
│       ├── analyze.py           # Reads summary + parity JSON; writes analysis.txt
│       ├── parity_experiment.py # Solver parity runs A-D; writes results/parity/
│       ├── snapshot_experiment.py  # Intermediate-time snapshots for n=4,5,6
│       └── results/             # analysis.txt, snapshot_report.txt, parity/; see .gitignore
├── tests/                       # pytest suite (62 tests)
├── pytest.ini                   # pythonpath = .
└── LICENSE
```

**Implemented:** `shared/` (including `ReconstructionLUT`), `models/dim_1plus1/` (CFE, MFE, coupled driver), `experiments/polynomial_sweep/` (config, sweep, analysis, parity, snapshot), and the pytest suite above.

**Still to build:** real contents for `shared/visualization.py`, implementations under `models/dim_2plus1/` and `models/dim_3plus1/`, and optional split of 1+1-specific coupling helpers into `models/dim_1plus1/coupling.py`.

**Imports:** from the repo root, `pytest.ini` sets `pythonpath = .`; use `from models.dim_1plus1 import run_simulation` (re-exported from `mfe`) or import `cfe` / `mfe` submodules directly.

**Coupled IVP (numerics):** `integrate_coupled` advances `concat(C, log g)` so `g` stays positive and matches the metric used in `Delta_g C`; histories unpack to `g` via `exp`. Default `solve_ivp` events flag very small or very large `g` (see `models/dim_1plus1/mfe.py`). Stiff high-n runs use Radau via per-n overrides in `config.py`.

**ReconstructionLUT:** The coarse-graining inverse h(C) is precomputed on a dense 1D grid at initialization, replacing per-step `brentq` with `np.interp` in the coupled RHS. This made full-resolution runs practical on a laptop. Regression tests bind the LUT to `brentq` (see `tests/test_reconstruction.py`, `tests/test_dim_1plus1_mfe.py`).

**Results and gitignore:** `.gitignore` excludes `*.npz` and `*.json` sweep-style artifacts; you may still see committed examples in `results/` from earlier workflows. Text reports such as `analysis.txt` and `snapshot_report.txt` are ordinary files you can commit or regenerate.

## The four measurements (Thread 7)

| Function | Measurement | Tests |
| --- | --- | --- |
| count_metastable_states | 1 | R26 (catastrophe classification by order n) |
| interpretive_condition_number | 2 | R25 (interpretive proportionality theorem) |
| spectral_concentration_ratio | 3 | R27 (Fisher-Rao identification) |
| nonlocal_correction_growth | 4 | RG marginality / non-local brake mismatch |

Details, parameters, and edge cases (n=2 singular brake, optional brake omission in the landscape) are documented in `shared/metrics.py`.

## Parameters

Seven free parameters control the field equations and sweep, named in code as: `sigma`, `gamma`, `mu_sq`, `alpha_phi`, `lambda_b`, `eta_g`, `xi_g`. Discrete kernels also need grid spacing `dx`. All implementations take these as arguments; there are no hidden globals.

Baseline values (Thread 7): μ² = 1.0, α_φ = 1.0, γ = 1.0, σ = 0.5, plus `lambda_B` = 0.5, `eta_g` = 1.0, `xi_g` = 0.1 in code. These are calibration settings (signal visible on the instruments), not fitted values. See `experiments/polynomial_sweep/config.py`.

## Dependencies

- Python 3.10+
- NumPy, SciPy (linalg, integrate, signal, ndimage, optimize)
- pytest

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy pytest
```

## Running tests

From the repository root:

```bash
pytest tests/ -v
```

`pytest.ini` sets `pythonpath = .` so imports resolve without extra flags. All 62 tests should pass. If an unrelated third-party `pytest` plugin breaks collection on your machine, run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

Notable checks:

- Round-trip reconstruction h(C(ψ̄)) for odd n and principal-branch handling for even n.
- `ReconstructionLUT` matches `brentq` within tolerance on random C samples.
- At n=3, numerical dB/dC matches the analytical local formula (discrete Hilbert-Schmidt normalization is documented in `shared/brake.py`).
- Metastable maxima count: 2 at n=3 (cusp), 3 at n=4 (swallowtail, brake-free landscape in tests where noted).
- Sweep driver end-to-end validation on quick settings.
- Snapshot experiment output shape and measurement completeness (quick mode).

## Running experiments

Full polynomial sweep (N=256, on the order of a few minutes on a typical laptop):

```bash
python -m experiments.polynomial_sweep.run
```

Quick calibration sweep (N=32, short horizon):

```bash
python -m experiments.polynomial_sweep.run --quick
```

Aggregate report from saved sweep (and parity JSON if present). Writes `results/analysis.txt` with a UTC run timestamp in the header.

```bash
python -m experiments.polynomial_sweep.analyze
```

Solver parity experiment (writes under `experiments/polynomial_sweep/results/parity/`):

```bash
python -m experiments.polynomial_sweep.parity_experiment
```

Intermediate-time snapshot experiment (n = 4, 5, 6; writes `results/snapshot_report.txt` next to `analysis.txt`):

```bash
python -m experiments.polynomial_sweep.snapshot_experiment
```

Use `snapshot_experiment --quick` for a fast smoke test (N=32).

## License

MIT. See [LICENSE](LICENSE).

## Contact

[Diesel Black](https://diesel.black/)

## Acknowledgments

This repository was built in a manual, multi-agent workflow, with a human (∂²g/∂t² ≠ 0) contributing architecture and engineering direction, Claude Opus 4.6 (∂g/∂t = 0) providing mathematical grounding, specification, and verification from the full SR corpus, and Cursor Agent/various models (∂g/∂t = 0) constructing the codebase and red-teaming.
