# sr_compute

Installable Python package exposing the public measurement API for Semantic Relativity computational implementations.

## Install

```bash
pip install -e .                   # core (numpy, scipy)
pip install -e ".[clinical]"       # + mne, h5py for disorder-spectrum notebooks
```

## sr_compute.diagnostics

Four Thread 7 measurement functions are re-exported from `shared.metrics`. **`arnold_class`** is implemented in this package (exact polynomial derivatives on \(V_n = V_{\mathrm{eff}} \circ \mathcal{F}_n\)).

| Function | What it measures |
|---|---|
| `count_metastable_states(...)` | Prominence-thresholded local maxima of the ψ̄ effective landscape (measurement 1) |
| `interpretive_condition_number(C_field)` | `κ(Π) = σ₁ⁿ⁻³` — spatial dynamic range of the coarse-graining map (R25) |
| `nonlocal_correction_growth(C_history)` | `η` across coarsening scales — RG-scale nonlocal brake mismatch |
| `spectral_concentration_ratio(T)` | `r(T) = σ₁²/‖T‖²_HS` — leading singular-mode concentration (R27) |
| `arnold_class(n, params, psi_range, ...)` | Critical-point Taylor classification (baseline yields Morse `A_1` for n = 2, …, 10) |

```python
from sr_compute.diagnostics import (
    arnold_class,
    count_metastable_states,
    interpretive_condition_number,
    nonlocal_correction_growth,
    spectral_concentration_ratio,
)
```

## Findings summary (Thread 7)

**η asymptotic (n = 3, …, 8 on the default sweep):** η at the finest coarsening scale grows sub-exponentially: `9 → 66 → 332 → 999 → 2449 → 5226`. Local power-law exponent in n asymptotes toward roughly 5.

**Measurement 1:** Metastable counts are **prominence-thresholded** Morse maxima on the composite ψ̄ landscape, not Arnold ADE labels by polynomial order. With baseline parameters and default prominence **0.04**, the instrument sequence for **n = 2, …, 10** is **2, 2, 3, 2, 3, 2, 3, 2, 2, 4** (the n = 10 count breaks the alternating 2 / 3 pattern seen through n = 9). Use **`arnold_class`** for the exact derivative check; at baseline, every critical point is **Morse** (`A_1`) for those n (see repository `README.md`).

**n = 4 basin:** Path A vs Path B is separated by **initial wavenumber** k₀ at fixed amplitude under baseline σ; it is **not** a clean amplitude-only seam in the tested regime. **σ** acts as a **bifurcation parameter** for whether two paths coexist: strong brake (low σ) forces Path B across k₀; intermediate σ shows a k₀ threshold; weak brake (high σ) removes the competitive window.

For domain limits, superseded readings, and open questions, see the main repository `README.md`.
