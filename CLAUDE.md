# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Run from the repo root (`/Users/ilin/Wake-T`). Python is managed via pyenv — use the alias `pyrun` (set in `~/.zshrc`) or the full path `~/.pyenv/versions/3.11.11/bin/python`.

```bash
# Install in development mode with test dependencies
pyrun -m pip install -e ".[test]"

# Run full test suite
pyrun -m pytest

# Run a single test file
pyrun -m pytest tests/test_adaptive_grid.py -v

# Run a specific test
pyrun -m pytest tests/test_salame.py::test_salame -v

# Lint and format
ruff check wake_t/
ruff format wake_t/
```

## Architecture

### Beamline framework

A simulation is built as a `Beamline` (or single `FieldElement`) composed of elements such as `PlasmaStage`, `Drift`, `Quadrupole`, etc. Tracking proceeds element by element:

```
Beamline.track(bunches) → FieldElement.track() → Tracker → field._calculate_wakefield() per step
```

- `wake_t/beamline_elements/` — `PlasmaStage`, `Drift`, magnets, `Beamline`, `FieldElement` base
- `wake_t/tracking/` — `Tracker` class that advances particles and calls diagnostics
- `wake_t/fields/` — field base classes; `RZWakefield` is the base for cylindrical plasma models
- `wake_t/particles/particle_bunch.py` — `ParticleBunch`: the core particle container
- `wake_t/diagnostics/openpmd_diag.py` — openPMD HDF5 output

### Plasma wakefield models

`PlasmaStage(wakefield_model=...)` maps string names to classes in `plasma_stage.py`:

| String | Class |
|--------|-------|
| `"quasistatic_2d"` / `"quasistatic_2d_ion"` | `Quasistatic2DWakefieldIon` (this subdir) |
| `"quasistatic_2d_legacy"` | `Quasistatic2DWakefield` (original, electrons only) |
| `"simple_blowout"` | analytical blowout model |

The ion model (`qs_rz_baxevanis_ion/`) is the active development target. See its own `CLAUDE.md` for details.

### openPMD dependency

`openpmd-api` is pinned to `<0.17.0` in `pyproject.toml`. Version 0.17.0 introduced strict homogeneous-extents validation that is incompatible with the `make_constant` approach used by `aptools` (a Wake-T dependency). Do not remove this pin without also updating `openpmd_diag.py`, `bunch_saving.py`, and the read path in `bunch_generation.py`.
