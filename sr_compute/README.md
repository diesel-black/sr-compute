# sr_compute

Installable Python package exposing the public measurement API for Semantic Relativity computational implementations.

## Install

```bash
pip install -e .                   # core (numpy, scipy)
pip install -e ".[clinical]"       # + mne, h5py for disorder-spectrum notebooks
```

## sr_compute.diagnostics

Four Thread 7 measurement functions, re-exported from `shared.metrics`:

| Function | What it measures |
|---|---|
| `count_metastable_states(psi_bar)` | Number of local minima of `V(ψ̄) = ψ̄²/2 + γψ̄ⁿ/(n·(n-1))` |
| `interpretive_condition_number(C_field)` | `κ(Π) = σ₁ⁿ⁻³` — spatial dynamic range of the coarse-graining map |
| `nonlocal_correction_growth(C_history)` | `η₂/η₁` — RG-scale growth of the nonlocal correction |
| `spectral_concentration_ratio(T)` | `r(T) = σ₁²/‖T‖²_HS` — leading singular-mode concentration |

```python
from sr_compute.diagnostics import (
    count_metastable_states,
    interpretive_condition_number,
    nonlocal_correction_growth,
    spectral_concentration_ratio,
)
```

## Findings summary (Thread 7, n=2..8)

**η asymptotic**: `r(T)` at `f=1` grows sub-exponentially with polynomial order:
`9 → 66 → 332 → 999 → 2449 → 5226` (n=3..8). Local power-law exponent asymptotes toward ~5.

**Parity law**: Metastable count is exactly 2 for odd n and 3 for even n across n=2..8.
Corresponds to Arnold ADE classification (cusp/swallowtail/butterfly alternation).

**n=4 domain limit**: The even-n coarse-graining map has a bounded valid domain (`C ≥ C_floor ≈ −0.47` at γ=1).
Large-amplitude ICs diverge; the basin bracket for the swallowtail boundary is `[5.2×10⁻², 1.9×10⁻¹]`.
