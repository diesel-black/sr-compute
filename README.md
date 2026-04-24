# sr-compute

Computational implementations of the Semantic Relativity (SR) toy models.

[Semantic Relativity](https://github.com/diesel-black/semantic-relativity) is a geometric field theory for observer-dependent meaning, formalizing how coherence, curvature, and coupling interact on semantic manifolds. This repository implements the toy models from Appendix A of the main paper as numerical simulations. It is scientific instrumentation: each module matches a named object in the formalism, not a generic PDE or ML stack.

For the full theory, axioms, Lagrangian, and dimensional roadmap, see the paper and Appendix A.

## Current focus

"Thread 7" (Cubic Aperture): a polynomial-order sweep n = 2, ..., 10 in the 1+1 model by default (`experiments/polynomial_sweep/config.py`), with four simultaneous measurements testing structural results R25 and R27, RG marginality probes, and measurement 1 (prominence-thresholded landscape maximum count). The experiment specification lives under `experiments/polynomial_sweep/`; the shared mathematics lives under `shared/`.

## Key findings

The polynomial sweep, solver-parity controls, intermediate-time snapshot, robustness, ensemble, and basin experiments together characterize the cubic aperture as a sharp dynamical boundary.

**Confirmed predictions:**

- **Interpretive proportionality (R25):** κ(Π) = 1.0 exactly at n = 3 (algebraic identity, exponent n−3 = 0). At n = 4, κ ≈ 3.98 on the sweep final state, confirming supercubic amplification of spatial dynamic range when the field is structured.
- **Composite landscape geometry:** At baseline SR parameters, every critical point of the pullback effective landscape along the ψ̄ axis used in measurement 1 is **Morse** (non-degenerate) for polynomial orders tested through **n = 10**. Check with `arnold_class` in `sr_compute.diagnostics` (exact polynomial derivatives) and the table driver `python -m experiments.polynomial_sweep.arnold_classification_report`, which writes `results/arnold_classification.txt`. This replaces the earlier Arnold ADE-by-n reading: there are no degenerate catastrophe points in that composite landscape at baseline, so polynomial order does not label an A₂, A₃, … tower here.
- **Measurement 1 (landscape maximum count):** `count_metastable_states` in `shared/metrics.py` counts **prominence-thresholded** local maxima (defaults `peak_prominence=0.04`, `peak_distance=120`). It is an instrument readout, not a structural theorem. With baseline parameters and that default prominence, the observed sequence for **n = 2, …, 10** is:

| n | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| metastable count | 2 | 2 | 3 | 2 | 3 | 2 | 3 | 2 | 2 | 4 |

The **n = 10** entry (four maxima) breaks the alternating 2 / 3 pattern seen through n = 9 when an inner fold-point maximum crosses the prominence threshold. Re-run `python -m experiments.polynomial_sweep.run` to refresh `summary.json` after changing `N_VALUES` or prominence defaults.

- **Cross-n non-locality scaling:** η grows as 9 → 66 → 332 → 999 from n = 3 through n = 6 on the sweep snapshots, confirming progressive degradation of the convolution-deconvolution cancellation (R3) with polynomial order (field-dependent magnitudes; interpret alongside integration time and spatial structure).
- **n = 4 simultaneous breakpoint:** κ ≈ 3.98, spectral ratio ≈ 0.519, and the only coarsening growth rate > 1 (1.197) in the sweep dataset. Three independent channels spike at the same polynomial order.

**Dynamical regime transition (n=4 | n=5 boundary):**

The snapshot experiment (`snapshot_experiment.py`, n = 4, 5, 6) plus the main sweep for n = 3 resolved the n ≥ 5 homogeneity question. At t ≈ 8, n = 4, 5, 6 show comparable coherence variation (C_range about 0.14 to 0.19 in the full-resolution `snapshot_report.txt`). They then diverge:

- **n = 3 and n = 4 (sweep):** the metric hits a terminal event (metric floor or step failure near the same window, §A.1.8), so the evolved state stays spatially structured at the last recorded time.
- **n = 5 and n = 6 (sweep + snapshots):** the integration reaches t = 30 without that termination; coherence relaxes toward spatial homogeneity between t ≈ 8 and t ≈ 15.

The cubic aperture boundary is not "which n develops structure" (they all do through the critical window) but "which n's metric survives the structured period without the same terminal geometry" (n ≥ 5 in this baseline). The non-saturating brake dynamics at supercubic orders are a leading hypothesis for that stabilization.

Solver-parity controls (`parity_experiment.py`) support treating these regime differences as physical rather than a Radau artifact: Radau and RK45 agree where both complete at n = 3 and n = 4; n = 5 with RK45 is stiff and may not finish, consistent with different dynamics rather than implicit damping inventing homogeneity.

**Extended sweep n = 2..8 (new results):**

The η asymptotic is now pinned through n = 8: **9 → 66 → 332 → 999 → 2449 → 5226** for n = 3, 4, 5, 6, 7, 8. The growth is sub-exponential: successive ratios are 7.3, 5.0, 3.0, 2.45, 2.13, monotonically decreasing. The local power-law exponent asymptotes toward approximately 5 rather than diverging. This is a direct empirical bound on the nonlocal-correction cost past the cubic aperture: it grows as a finite power of n, not factorially. The cubic seam remains the unique aperture because η₃ = 9 is the only value small enough for the brake to hold without truncation.

**n = 4 bimodal basin (full-resolution path, N = 256):** Path assignment is not driven by IC amplitude in the tested low-amplitude regime: for fixed seeds, Path A vs Path B is amplitude-independent; large amplitudes still hit the even-n principal-branch floor (divergence). The discriminator is initial wavenumber k₀ at fixed A = 10⁻² and baseline σ = 0.5: Path B (amplitude death) for k₀ = 1, …, 7 and Path A (fine-structure death) for k₀ = 8, …, 10, with a clean boundary between k₀ = 7 and k₀ = 8. IC power spectra across seeds do not explain the seed-wise Path A / Path B split (hypothesis ruled out on the instrument).

**σ as bifurcation parameter for bimodality:** In this implementation, **σ** rescales brake strength and MFE coefficients (for example ζ_cubic(γ, σ) ∝ σ⁻³) as a **scalar** prefactor; it does not act as a spatial convolution width on the IVP. A coupling-scale sweep still shows that **two-path coexistence is not generic** at n = 4: at σ = 0.3 every tested k₀ stays Path B; at σ = 0.5 the k₀ threshold appears; at σ = 1.0 runs collapse toward Path A behavior without the same competitive window. Neither k₀^crit · σ ≈ const nor σ-independent k₀^crit holds across those regimes. The open structure is a **window in σ** (lower edge between 0.3 and 0.5, upper edge near 1.0) inside which (σ, k₀) admits a genuine bifurcation diagram.

**Superseded narrative (do not re-import):** Arnold ADE classification by polynomial order; metastable **parity as a law**; swallowtail **codimension-1 boundary primarily in amplitude** A; **kernel physical resolution** as the explanation of the k₀ ≈ 7.5 threshold.

**Open questions:**

- Map the **σ-window** edges and the two-dimensional (σ, k₀) bifurcation set; classify boundary bifurcations if possible (here catastrophe language applies to **parameter** space, not to labeling by n).
- **Seed-to-path** mechanism at n = 4 when amplitude and power spectrum do not discriminate (phase or higher-order functionals of the IC).
- Why does the brake's lack of a saturation maximum at n ≥ 4 relate to avoidance of the metric terminal event? The connection between R4 (brake saturation at |ψ̄| = 1/√(3γ)) and §A.1.8 needs analytical investigation.
- Parameter stress-testing: do the qualitative findings hold under varied μ², γ, σ? (Empirical check: `robustness_experiment.py` and `results/robustness_report.txt`.)
- What is the precise asymptotic exponent of the η power law as n → ∞?
- **Metastable count beyond n = 10:** further fold pairs, saturation, or non-monotone behavior under a fixed prominence rule.

## Theory in brief

SR couples a coherence field C(x,t) and a metric g(x,t) on a semantic manifold through three field equations:

- **CFE** (Coherence Field Equation): how coherence evolves given the geometry.
- **RFE** (Recurgent Field Equation): how coherence sources curvature (trivial in 1D since R = 0).
- **MFE** (Metric Field Equation): how the metric evolves given coherence structure.

For the 1+1 implementation:

- Coarse-graining at polynomial order n: C = ψ̄ + γₙ ψ̄ⁿ (in code: `psi_bar`, `gamma`, and integer `n`).
- Effective potential: V_eff(C) = (μ²/2) C² − α_φ C⁴ (attractor vs autopoietic terms; parameter `alpha_phi` in code).
- The CFE reaction term is +V′_eff(C), so the system ascends V_eff. Stable equilibria are at the local maxima of V_eff (at ±C*), while the origin is unstable. Measurement 1 counts **prominence-thresholded** local maxima of the ψ̄-space effective landscape in `shared/metrics.py` (not a catastrophe-class label by n).
- The coupling tensor K and brake functional B[K] discretize Appendix A sections A.1.2 through A.1.3. Numerical vs analytical brake variation is calibrated at n=3 and diverges for n>3 as tested in `tests/test_brake.py`.

## Repository layout

```
sr-compute/
├── sr_compute/                  # Installable public API (pip install -e .)
│   ├── __init__.py
│   └── diagnostics.py           # r(T), κ, η, measurement 1, arnold_class (exact V_n critical points)
├── shared/                      # Math primitives (general n), heavily tested
│   ├── __init__.py
│   ├── potentials.py            # V, Phi, V_eff, A(C), landscapes + brake quadrature
│   ├── reconstruction.py        # F_n(psi_bar), dC/d(psi_bar), h(C), ReconstructionLUT
│   ├── coupling.py              # Gaussian kernel, kappa_n, periodic K matrix
│   ├── brake.py                 # zeta (cubic), analytical vs numerical dB/dC, saturation scale
│   ├── metrics.py               # R25, R27, RG marginality, measurement 1 (landscape count)
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
│       ├── config.py            # Baseline parameters, grid specs, per-n solver overrides (n=2..10)
│       ├── run.py               # Sweep driver (--quick, --n, --no-save, optional --wallclock)
│       ├── analyze.py           # Reads summary + parity JSON; writes analysis.txt (dynamic over n)
│       ├── outcome_utils.py     # IVP outcome labels (completed / terminal / timeout) for reports
│       ├── parity_experiment.py # Solver parity runs A-D; writes results/parity/
│       ├── snapshot_experiment.py        # Intermediate-time snapshots for n=4,5,6
│       ├── robustness_experiment.py      # Single-parameter stress test, n=3,4,5
│       ├── ensemble_experiment.py        # Multi-seed n=4 + n=3 control; optional wallclock per run
│       ├── bimodal_basin_experiment.py   # IC amplitude + wavenumber sweep to map n=4 basin boundary
│       ├── arnold_classification_report.py  # Tabulate arnold_class over n -> results/arnold_classification.txt
│       ├── coupling_scale_experiment.py  # Seed spectra + σ sweep for k0 crit at n=4
│       └── results/             # analysis.txt, reports, parity/; see .gitignore
├── tests/                       # pytest suite (90+ tests)
├── pyproject.toml               # pip install -e . for sr_compute package
├── pytest.ini                   # pythonpath = .
└── LICENSE
```

**Implemented:** `shared/` (including `ReconstructionLUT`), `models/dim_1plus1/` (CFE, MFE, coupled driver), `experiments/polynomial_sweep/` (config, sweep, analysis, parity, snapshot, robustness, ensemble, basin, shared `outcome_utils`), `sr_compute/diagnostics` (installable public API), and the pytest suite.

**Still to build:** real contents for `shared/visualization.py`, implementations under `models/dim_2plus1/` and `models/dim_3plus1/`, and optional split of 1+1-specific coupling helpers into `models/dim_1plus1/coupling.py`.

**Imports:** from the repo root, `pytest.ini` sets `pythonpath = .`; use `from models.dim_1plus1 import run_simulation` (re-exported from `mfe`) or import `cfe` / `mfe` submodules directly. External notebooks use `from sr_compute.diagnostics import spectral_concentration_ratio` or `arnold_class` after `pip install -e .`.

**Coupled IVP (numerics):** `integrate_coupled` advances `concat(C, log g)` so `g` stays positive and matches the metric used in `Delta_g C`; histories unpack to `g` via `exp`. Default `solve_ivp` events flag very small or very large `g` (see `models/dim_1plus1/mfe.py`). Stiff high-n runs use Radau via per-n overrides in `config.py`.

**ReconstructionLUT:** The coarse-graining inverse h(C) is precomputed on a dense 1D grid at initialization, replacing per-step `brentq` with `np.interp` in the coupled RHS. This made full-resolution runs practical on a laptop. Regression tests bind the LUT to `brentq` (see `tests/test_reconstruction.py`, `tests/test_dim_1plus1_mfe.py`). For even n, the LUT has a C_floor below which h(C) is undefined (principal-branch limit); the basin experiment exposes this as a physical domain constraint on IC amplitude.

**Results and gitignore:** `.gitignore` excludes `*.npz` and `*.json` sweep-style artifacts; you may still see committed examples in `results/` from earlier workflows. Text reports such as `analysis.txt` and `snapshot_report.txt` are ordinary files you can commit or regenerate.

## The four measurements (Thread 7)

| Function | Measurement | Tests |
| --- | --- | --- |
| count_metastable_states | 1 | Prominence-thresholded maxima on the ψ̄ landscape |
| interpretive_condition_number | 2 | R25 (interpretive proportionality theorem) |
| spectral_concentration_ratio | 3 | R27 (Fisher-Rao identification) |
| nonlocal_correction_growth | 4 | RG marginality / non-local brake mismatch |

All four are re-exported from `sr_compute.diagnostics` for use in disorder-spectrum and clinical notebooks. **`arnold_class`** lives in the same module and is not one of the four sweep measurements; it classifies critical points of \(V_n = V_{\mathrm{eff}} \circ \mathcal{F}_n\) by exact Taylor coefficients. Details, parameters, and edge cases (n=2 singular brake, optional brake omission in the landscape) are documented in `shared/metrics.py` and `sr_compute/diagnostics.py`.

## Parameters

Seven free parameters control the field equations and sweep, named in code as: `sigma`, `gamma`, `mu_sq`, `alpha_phi`, `lambda_b`, `eta_g`, `xi_g`. Discrete kernels also need grid spacing `dx`. All implementations take these as arguments; there are no hidden globals.

Baseline values (Thread 7): μ² = 1.0, α_φ = 1.0, γ = 1.0, σ = 0.5, plus `lambda_B` = 0.5, `eta_g` = 1.0, `xi_g` = 0.1 in code. These are calibration settings (signal visible on the instruments), not fitted values. See `experiments/polynomial_sweep/config.py`.

## Dependencies

- Python 3.10+
- NumPy, SciPy (linalg, integrate, signal, ndimage, optimize)
- pytest

Install sweep environment and `sr_compute` package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy pytest
pip install -e .          # installs sr_compute.diagnostics as an importable package
```

Optional clinical/disorder-spectrum dependencies (MNE, HDF5):

```bash
pip install -e ".[clinical]"
```

## Running tests

From the repository root:

```bash
pytest tests/ -v
```

`pytest.ini` sets `pythonpath = .` so imports resolve without extra flags. All tests in `tests/` should pass (90+ as of the Arnold diagnostic merge). If an unrelated third-party `pytest` plugin breaks collection on your machine, run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

Notable checks:

- Round-trip reconstruction h(C(ψ̄)) for odd n and principal-branch handling for even n.
- `ReconstructionLUT` matches `brentq` within tolerance on random C samples.
- At n=3, numerical dB/dC matches the analytical local formula (discrete Hilbert-Schmidt normalization is documented in `shared/brake.py`).
- Metastable maxima count: baseline tests expect two maxima at n=3 and three at n=4 under `count_metastable_states` defaults (see `tests/test_metrics.py`).
- `arnold_class` returns Morse (`A_1`) critical-point data at baseline parameters for n = 2, …, 10 (`tests/test_arnold_class.py`).
- Sweep driver end-to-end validation on quick settings (`test_sweep_driver.py`): `run_single` / `run_sweep` / save-load round-trip; saved `n*_measurements.json` includes an `outcome` field (`completed`, `terminal`, or `timeout`).
- Snapshot experiment output shape and measurement completeness (quick mode, `test_snapshot_experiment.py`).
- Robustness experiment (`test_robustness_experiment.py`): quick mode runs baseline plus two perturbations for n=3,4,5; all table columns finite; row keys use `outcome` (not raw SciPy `success`).
- Ensemble experiment (`test_ensemble_experiment.py`): quick mode covers all seeds for n=4 and n=3; report contains usable-run statistics and seed-diversity lines; n=3 control rows satisfy κ≈1; optional Unix-only test for immediate wallclock timeout rows (NaN measurements, `outcome=timeout`).

## Running experiments

Full polynomial sweep (N=256, n=2..10 by default, on the order of several minutes on a typical laptop):

```bash
python -m experiments.polynomial_sweep.run
```

Quick calibration sweep (N=32, short horizon):

```bash
python -m experiments.polynomial_sweep.run --quick
```

Run specific n values only:

```bash
python -m experiments.polynomial_sweep.run --n 7 8
```

Aggregate report from saved sweep (and parity JSON if present). Writes `results/analysis.txt` with a UTC run timestamp in the header. Table columns auto-expand to include all n in `summary.json`:

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
python -m experiments.polynomial_sweep.snapshot_experiment --quick
```

Parameter robustness (baseline plus 12 single-parameter perturbations, n=3,4,5; writes `results/robustness_report.txt`):

```bash
python -m experiments.polynomial_sweep.robustness_experiment
python -m experiments.polynomial_sweep.robustness_experiment --quick
```

Multi-seed ensemble at n=4 with n=3 control (writes `results/ensemble_report.txt`; full mode uses per-run wallclock caps, suppresses known `ReconstructionLUT` warnings in the CLI entrypoint only):

```bash
python -m experiments.polynomial_sweep.ensemble_experiment
python -m experiments.polynomial_sweep.ensemble_experiment --quick
```

Bimodal basin characterization at n=4: IC amplitude sweep (Phase 1) + sinusoidal wavenumber sweep (Phase 2). Full-resolution runs show Path A vs Path B separated by **k₀** at fixed amplitude (not by amplitude alone) and expose the even-n principal-branch domain limit at large amplitude. Writes `results/bimodal_basin_report.txt`:

```bash
python -m experiments.polynomial_sweep.bimodal_basin_experiment
python -m experiments.polynomial_sweep.bimodal_basin_experiment --quick
```

Symbolic Arnold-class table over n (writes `results/arnold_classification.txt`):

```bash
python -m experiments.polynomial_sweep.arnold_classification_report
```

Coupling-scale experiment: IC power spectra for basin seeds plus σ sweep for k₀ discrimination at n=4 (writes `results/coupling_scale_report.txt`):

```bash
python -m experiments.polynomial_sweep.coupling_scale_experiment
python -m experiments.polynomial_sweep.coupling_scale_experiment --quick
```

## License

MIT. See [LICENSE](LICENSE).

## Contact

[Diesel Black](https://diesel.black/)

## Acknowledgments

This repository was built in a manual, multi-agent workflow, with a human (∂²g/∂t² ≠ 0) contributing architecture and engineering direction, Claude Opus 4.6 (∂g/∂t = 0) providing mathematical grounding, specification, and verification from the full SR corpus, and Cursor Agent/Claude Code/various models (∂g/∂t = 0) constructing the codebase and red-teaming.
