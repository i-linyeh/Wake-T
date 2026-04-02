# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This subdirectory (`qs_rz_baxevanis_ion`) implements a quasi-static 2D plasma wakefield solver in cylindrical (r-z) geometry that supports ion motion, laser drivers, adaptive grids, and the SALAME beam-loading shaping algorithm — extending the original electron-only `qs_rz_baxevanis` model.

## Commands

All commands run from the Wake-T root (`/Users/ilin/Wake-T`). Use `pyrun` alias or `~/.pyenv/versions/3.11.11/bin/python`.

```bash
pyrun -m pip install -e ".[test]"
pyrun -m pytest tests/test_adaptive_grid.py -v
pyrun -m pytest tests/test_salame.py -v
ruff check wake_t/
ruff format wake_t/
```

## Architecture

### Data flow (per time step)

```
Quasistatic2DWakefieldIon._calculate_wakefield(bunches)
  ├─ [SALAME IC, first step only if witness.do_salame=True]
  │   └─ beamloading_initial_condition()  →  updates witness bunch weights
  ├─ deposit_bunch_charge() / AdaptiveGrid.calculate_bunch_source()
  ├─ calculate_wakefields()  →  solver.py  (Numba JIT, gridless radial columns)
  │   ├─ _normalize_grid() / _setup_laser()  (shared setup helpers)
  │   ├─ b_theta.py / psi_and_derivatives.py  (analytic field kernels)
  │   └─ plasma_particles.py + PlasmaParticleContainer
  └─ gather_main_fields_cyl_linear()  →  fields → bunch particles
```

### Module roles

- `wakefield.py` — top-level class `Quasistatic2DWakefieldIon`; owns the base grid, adaptive grids, SALAME flags, and the main `_calculate_wakefield` loop
- `solver.py` — core Numba-JIT loop (`evolve_one_step`); also contains SALAME helper functions (`build_pp_cache_at_kp1`, `commit_cache_one_slice`, `calculate_wakefields_ez_km1_from_cache`) and shared setup helpers (`_normalize_grid`, `_setup_laser`)
- `plasma_particles.py` + `plasma_particle_container.py` — plasma particle storage; container is a plain-data struct that crosses the Python/Numba boundary
- `adaptive_grid.py` — `AdaptiveGrid`: adjusts radial/longitudinal extents to the bunch each step
- `beamloading_initial_condition.py` — SALAME algorithm; modifies `witness.w` in-place on the first wakefield solve
- `b_theta_bunch.py` — azimuthal B-field source term from particle bunches; `calculate_bunch_source` (full grid) and `calculate_bunch_source_slice` (single slice, used by SALAME)
- `deposition.py` / `gather.py` — charge deposition and field gathering on the base grid
- `psi_and_derivatives.py` / `b_theta.py` — analytic kernel evaluations (all `@njit_serial`)
- `utils.py` — gradient utilities, chi/rho calculation, laser envelope helpers

### Performance pattern

All hot-path functions use `@njit_serial()` from `wake_t.utilities.numba`. `PlasmaParticleContainer` serialises particle arrays into a plain tuple so they can cross the Python → Numba boundary. When editing inner-loop code, keep this pattern — Numba functions must not receive arbitrary Python objects.

`evolve_one_step` accepts `start_slice_i` / `stop_slice_i` so it can be called on a sub-window of slices, enabling the inline SALAME design (see below).

### Shared setup helpers in solver.py

`_normalize_grid(r_max, xi_min, xi_max, n_r, n_xi, n_p, r_fld)` → `(s_d, dr, dxi, r_fld_n)`
`_setup_laser(laser_a2, dr, n_xi, n_r)` → `(laser_a2, nabla_a2, has_laser_source)`

Used by `calculate_wakefields`, `build_pp_cache_at_kp1`, `commit_cache_one_slice`, and `calculate_wakefields_ez_km1_from_cache` to avoid repeating normalization and laser setup.

### Relation to `qs_rz_baxevanis`

`qs_rz_baxevanis/` is the original electron-only model. The two directories are conceptually related but not interdependent. `"quasistatic_2d"` in `plasma_stage.py` maps to `Quasistatic2DWakefieldIon` (this module), not the legacy version.

## SALAME interface

SALAME shapes the witness-bunch charge profile to flatten the accelerating field. It only runs on the **first** call to `_calculate_wakefield` (guarded by `self._initial_condition_done`).

**Enable by setting attributes directly on the witness `ParticleBunch`:**
```python
witness.do_salame = True
witness.salame_n_iter = 100               # bisection iterations per slice (default 10)
witness.salame_relative_tolerance = 1e-6  # convergence tolerance (default 1e-4)
```

**Witness bunch selection (`_select_witness_bunch`):**
1. First bunch with `do_salame=True` attribute set (highest priority)
2. Last bunch in the list (fallback for beam-driven case)

**Key solver functions used by SALAME** (in `solver.py`):
- `build_pp_cache_at_kp1` — initializes plasma and advances to just before the witness tail
- `commit_cache_one_slice` — extends the cache one slice at a time as SALAME sweeps head-ward
- `calculate_wakefields_ez_km1_from_cache` — evaluates Ez[k-1] via centered stencil `-(ψ[k]-ψ[k-2])/(2dξ)` without mutating the cache (deepcopies pp_state internally)

**2dξ oscillation in SALAME current:** the centered Ez stencil couples only same-parity ψ values (k and k-2 share parity), while `b_t_bunch[k]` controls ψ[k-2] via the AB2 chain `b_t_bunch[k] → dpr[k] → pr[k-1] → dr[k-1] → r[k-2] → ψ[k-2]`. This decouples even and odd ψ chains, causing SALAME to converge to alternating g* values.

**Inline SALAME (base-grid only, pending implementation):** instead of a separate pre-pass, SALAME can be integrated into a single plasma evolution by calling `evolve_one_step` in three segments:
1. Phase 1 (JIT): `evolve_one_step(start=n_xi-1, stop=k_tail+2)` — pre-witness, fixed source
2. Phase 2 (Python): bisection per witness slice; each trial deepcopies pp_state and calls `evolve_one_step(start=k, stop=k-2)`; each commit calls `evolve_one_step(start=k, stop=k)`
3. Phase 3 (JIT): `evolve_one_step(start=k_head, stop=0)` — post-witness, shaped source

This eliminates the duplicate plasma initialization of the current two-pass approach and improves accuracy (~100× lower residual) because the bisection and the final field solve use the same plasma realization.

## Tests

- `tests/test_salame.py` — SALAME residual tests; `test_salame` uses adaptive grids, `test_salame_inline_vs_prestep` uses base grid only
- `tests/test_adaptive_grid.py` — end-to-end simulation with adaptive grid
- `tests/test_gradients.py` — custom gradient functions in `utils.py`
- `tests/test_sc_baxevanis.py` — space-charge calculations
- `tests/test_beam_deposition.py`, `tests/test_field_gathering.py`, `tests/test_plasma_deposition.py`
