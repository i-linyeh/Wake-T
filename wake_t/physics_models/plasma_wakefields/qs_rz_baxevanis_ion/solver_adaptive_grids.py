"""
Adaptive-grid helper solver for SALAME / beam-loading initial-condition
iteration in the Baxevanis-Stupakov quasi-static model.

Design goal
-----------
Keep the plasma evolution semantics identical to the working base-grid solver:
- build cache "ready for slice k"
- trial-evolve k..k-2 to get Ez(k-1)
- commit one fixed-source slice at a time

but project psi / b_theta onto a single target adaptive grid.

Important conventions
---------------------
- Plasma evolution and laser gathering always march on GLOBAL base-grid slice
  indices `slice_i`.
- `ag_i_grid` stores the GLOBAL slice indices covered by the target adaptive
  grid, with length `ag_nxi`.
- `ag_r_grid` contains the adaptive-grid radial nodes WITHOUT guard cells but
  INCLUDING the 2 border cells used by the AG machinery.
- `ag_psi_grid`, `ag_bt_grid` include 2 guard cells on each side and therefore
  have shape `(ag_nxi+4, ag_nr+4)`.

This file intentionally keeps the debug prints.
"""

from __future__ import annotations

import numpy as np
import scipy.constants as ct
import aptools.plasma_accel.general_equations as ge

from .plasma_particles import (
    pp_initialize,
    pp_sort,
    pp_gather_laser_sources,
    pp_gather_bunch_sources,
    pp_calculate_fields,
    pp_calculate_weights,
    pp_deposit_rho,
    pp_deposit_chi,
    pp_store_current_step,
    pp_evolve,
)
from .plasma_particle_container import PlasmaParticleContainer
from .psi_and_derivatives import calculate_psi_with_interpolation
from .b_theta import calculate_b_theta_with_interpolation
from .utils import radial_gradient
from wake_t.utilities.numba import njit_serial


def deepcopy_pp_state(pp_state):
    """
    Deep-copy serialized plasma state: tuple(species0_tuple, species1_tuple,...)
    Arrays are copied; scalars/bools are kept.
    """
    out = []
    for sp in pp_state:
        sp2 = []
        for item in sp:
            if hasattr(item, "copy"):
                sp2.append(item.copy())
            else:
                sp2.append(item)
        out.append(tuple(sp2))
    return tuple(out)


# ---------------------------------------------------------------------------
# AG projection helpers
# ---------------------------------------------------------------------------

@njit_serial
def pp_calculate_fields_on_single_ag_row(
    species_list,
    ag_r_grid_n,   # normalized by s_d, no guards, includes border cells
    ag_psi_row,    # no guards slice view
    ag_bt_row,     # no guards slice view
):
    """
    Project the CURRENT plasma-particle state onto one radial row of one
    adaptive grid.
    """
    ag_psi_row[:] = 0.0
    ag_bt_row[:] = 0.0

    add = False
    for s in species_list:
        calculate_psi_with_interpolation(
            r_eval=ag_r_grid_n,
            r=s.r,
            log_r=s.log_r,
            sum_1_arr=s.sum_1,
            sum_2_arr=s.sum_2,
            psi=ag_psi_row,
            add=add,
        )
        add = True

    for s in species_list:
        if not s.is_ion:
            calculate_b_theta_with_interpolation(
                r_fld=ag_r_grid_n,
                a_0=s.a_0[0],
                a=s.a_i,
                b=s.b_i,
                r=s.r,
                b_theta=ag_bt_row,
            )
            return


@njit_serial
def pp_project_fields_to_single_ag(
    species_list,
    slice_i,       # GLOBAL base-grid slice index
    ag_i_grid,     # GLOBAL slice indices covered by AG, length ag_nxi
    ag_r_grid_n,   # normalized by s_d, no guards, includes border cells
    ag_psi_grid,   # with guards
    ag_bt_grid,    # with guards
):
    """
    Write the CURRENT global slice `slice_i` onto the single target adaptive
    grid if that slice is covered by the AG.
    """
    if ag_i_grid.shape[0] == 0:
        return

    if slice_i < ag_i_grid[0] or slice_i > ag_i_grid[-1]:
        return

    # ag_i_grid is contiguous in practice. Keep the same mapping convention
    # as the rest of your AG code:
    #   local row in guarded AG array = (slice_i - ag_i_grid[0]) + 2
    local_i = slice_i + 2 - ag_i_grid[0]

    pp_calculate_fields_on_single_ag_row(
        species_list,
        ag_r_grid_n,
        ag_psi_grid[local_i, 2:-2],
        ag_bt_grid[local_i, 2:-2],
    )


# ---------------------------------------------------------------------------
# Window evolution kernels
# Split into nobeam / beam versions so Numba never has to type-check
# pp_gather_bunch_sources when there are no beam sources.
# ---------------------------------------------------------------------------

@njit_serial
def evolve_window_inplace_AG_nobeam(
    pp_serialized_list,
    n_xi,
    n_r,
    dxi,
    dr,
    r_fld_n,          # BASE-grid r, normalized
    has_laser_source,
    laser_a2,         # BASE-grid laser array with guards
    nabla_a2,         # BASE-grid radial gradient with guards
    max_gamma,

    # single target AG
    ag_i_grid,
    ag_r_grid_n,      # normalized, no guards, includes border cells
    ag_psi_grid,      # guarded AG output
    ag_bt_grid,       # guarded AG output

    shape,
    calculate_rho,
    rho,
    rho_e,
    rho_i,
    chi,
    store_plasma_history,
    particle_diags,
    start_slice_i,    # inclusive, GLOBAL index
    stop_slice_i,     # inclusive, GLOBAL index
):
    """
    Resume from existing plasma state and evolve only a small slice window.
    Write psi and b_theta only onto the single target adaptive grid.
    No beam-source gathering in this kernel.
    """
    print("In evolve_window_inplace_AG_nobeam, start")
    ions_computed = False
    pp_species_list = [PlasmaParticleContainer(species) for species in pp_serialized_list]

    # laser_a2 / nabla_a2 live on the BASE grid
    dr_base = r_fld_n[1] - r_fld_n[0]

    for slice_i in range(start_slice_i, stop_slice_i - 1, -1):
        pp_sort(pp_species_list)

        print("In evolve_window_inplace_AG_nobeam, start laser source")
        if has_laser_source:
            pp_gather_laser_sources(
                pp_species_list,
                laser_a2[slice_i + 2],
                nabla_a2[slice_i + 2],
                r_fld_n[0],
                r_fld_n[-1],
                dr_base,
            )

        pp_calculate_fields(pp_species_list, ions_computed, max_gamma)

        pp_project_fields_to_single_ag(
            pp_species_list,
            slice_i,
            ag_i_grid,
            ag_r_grid_n,
            ag_psi_grid,
            ag_bt_grid,
        )

        if calculate_rho:
            pp_deposit_rho(
                pp_species_list,
                ions_computed,
                shape,
                rho[slice_i + 2],
                rho_e[slice_i + 2],
                rho_i[slice_i + 2],
                r_fld_n,
                n_r,
                dr,
            )
        elif "w" in particle_diags:
            pp_calculate_weights(pp_species_list, ions_computed)

        if has_laser_source:
            pp_deposit_chi(
                pp_species_list,
                shape,
                chi[slice_i + 2],
                r_fld_n,
                n_r,
                dr,
            )

        ions_computed = True

        if store_plasma_history:
            pp_store_current_step(pp_species_list, particle_diags)

        if slice_i > 0:
            pp_evolve(pp_species_list, dxi)


@njit_serial
def evolve_window_inplace_AG_beam(
    pp_serialized_list,
    n_xi,
    n_r,
    dxi,
    dr,
    r_fld_n,              # BASE-grid r, normalized
    has_laser_source,
    laser_a2,             # BASE-grid laser array with guards
    nabla_a2,             # BASE-grid radial gradient with guards
    bunch_source_arrays,      # tuple of 2D arrays
    bunch_source_xi_indices,  # tuple of 1D int arrays of GLOBAL slice indices
    bunch_source_metadata,    # tuple of 1D float arrays [r_min, r_max, dr]
    max_gamma,

    # single target AG
    ag_i_grid,
    ag_r_grid_n,          # normalized, no guards, includes border cells
    ag_psi_grid,          # guarded AG output
    ag_bt_grid,           # guarded AG output

    shape,
    calculate_rho,
    rho,
    rho_e,
    rho_i,
    chi,
    store_plasma_history,
    particle_diags,
    start_slice_i,        # inclusive, GLOBAL index
    stop_slice_i,         # inclusive, GLOBAL index
):
    """
    Same as evolve_window_inplace_AG_nobeam but with bunch-source gathering.
    """
    print("In evolve_window_inplace_AG_beam, start")
    ions_computed = False
    pp_species_list = [PlasmaParticleContainer(species) for species in pp_serialized_list]

    dr_base = r_fld_n[1] - r_fld_n[0]

    for slice_i in range(start_slice_i, stop_slice_i - 1, -1):
        pp_sort(pp_species_list)

        print("In evolve_window_inplace_AG_beam, start laser source")
        if has_laser_source:
            pp_gather_laser_sources(
                pp_species_list,
                laser_a2[slice_i + 2],
                nabla_a2[slice_i + 2],
                r_fld_n[0],
                r_fld_n[-1],
                dr_base,
            )

        print("In evolve_window_inplace_AG_beam, start beam source")
        pp_gather_bunch_sources(
            pp_species_list,
            bunch_source_arrays,
            bunch_source_xi_indices,
            bunch_source_metadata,
            slice_i,
        )

        pp_calculate_fields(pp_species_list, ions_computed, max_gamma)

        pp_project_fields_to_single_ag(
            pp_species_list,
            slice_i,
            ag_i_grid,
            ag_r_grid_n,
            ag_psi_grid,
            ag_bt_grid,
        )

        if calculate_rho:
            pp_deposit_rho(
                pp_species_list,
                ions_computed,
                shape,
                rho[slice_i + 2],
                rho_e[slice_i + 2],
                rho_i[slice_i + 2],
                r_fld_n,
                n_r,
                dr,
            )
        elif "w" in particle_diags:
            pp_calculate_weights(pp_species_list, ions_computed)

        if has_laser_source:
            pp_deposit_chi(
                pp_species_list,
                shape,
                chi[slice_i + 2],
                r_fld_n,
                n_r,
                dr,
            )

        ions_computed = True

        if store_plasma_history:
            pp_store_current_step(pp_species_list, particle_diags)

        if slice_i > 0:
            pp_evolve(pp_species_list, dxi)


# ---------------------------------------------------------------------------
# Public helper functions used by beamloading_initial_condition_adaptive_grids.py
# ---------------------------------------------------------------------------

def _prepare_bunch_source_args(
    bunch_source_arrays,
    bunch_source_xi_indices,
    bunch_source_metadata,
):
    """
    Convert Python lists to Numba-friendly tuples with explicit dtypes.
    """
    if len(bunch_source_arrays) != len(bunch_source_xi_indices):
        raise ValueError("bunch_source_arrays and bunch_source_xi_indices length mismatch")
    if len(bunch_source_arrays) != len(bunch_source_metadata):
        raise ValueError("bunch_source_arrays and bunch_source_metadata length mismatch")

    arrays_arg = tuple(np.asarray(a, dtype=np.float64) for a in bunch_source_arrays)
    xi_idx_arg = tuple(np.asarray(idx, dtype=np.int64) for idx in bunch_source_xi_indices)
    meta_arg = tuple(np.asarray(md, dtype=np.float64) for md in bunch_source_metadata)
    return arrays_arg, xi_idx_arg, meta_arg


def build_pp_cache_at_kp1(
    k_tail,
    laser_a2,
    r_max,
    xi_min,
    xi_max,
    n_r,
    n_xi,
    ppc,
    n_p,
    r_max_plasma,
    radial_density,
    p_shape,
    max_gamma,
    plasma_pusher,
    ion_motion,
    ion_mass,
    free_electrons_per_ion,
    bunch_source_arrays,
    bunch_source_xi_indices,
    bunch_source_metadata,
    fld_arrays,

    # single target AG
    ag_i_grid,
    ag_r_grid,     # physical units, no guards, includes border cells
    ag_psi_grid,
    ag_bt_grid,
):
    """
    Build a plasma cache at slice k_tail+1, so the returned state is ready
    to evolve slice k_tail next.

    In SALAME single-AG mode, only the target AG is written.
    """
    print("In build_pp_chache, start")

    rho, rho_e, rho_i, chi, E_r, E_z, B_t, xi_fld, r_fld = fld_arrays

    # Normalize geometry for plasma evolution on the AG domain.
    s_d = ge.plasma_skin_depth(n_p * 1e-6)
    r_max_n = r_max / s_d
    xi_min_n = xi_min / s_d
    xi_max_n = xi_max / s_d
    dr = r_max_n / n_r
    dxi = (xi_max_n - xi_min_n) / (n_xi - 1)

    ppc_n = ppc.copy()
    ppc_n[:, 0] /= s_d

    r_max_plasma_n = r_max_plasma / s_d
    r_fld_n = r_fld / s_d
    ag_r_grid_n = ag_r_grid / s_d

    def radial_density_normalized(r):
        return radial_density(r * s_d) / n_p

    print("In build_pp_chache, after normalization")

    # Base-grid scratch arrays only
    B_t[:] = 0.0
    chi[:] = 0.0
    rho[:] = 0.0
    rho_e[:] = 0.0
    rho_i[:] = 0.0
    E_r[:] = 0.0
    E_z[:] = 0.0

    # Target AG scratch/output
    ag_psi_grid[:] = 0.0
    ag_bt_grid[:] = 0.0

    # Laser source: laser_a2 is on BASE grid
    has_laser_source = laser_a2 is not None
    if has_laser_source:
        n_xi_base = laser_a2.shape[0] - 4
        n_r_base = laser_a2.shape[1] - 4
        dr_base = r_fld_n[1] - r_fld_n[0]
        nabla_a2 = np.zeros((n_xi_base + 4, n_r_base + 4))
        radial_gradient(laser_a2[2:-2, 2:-2], dr_base, nabla_a2[2:-2, 2:-2])
    else:
        laser_a2 = np.zeros((0, 0))
        nabla_a2 = np.zeros((0, 0))

    arrays_arg, xi_idx_arg, meta_arg = _prepare_bunch_source_args(
        bunch_source_arrays,
        bunch_source_xi_indices,
        bunch_source_metadata,
    )
    has_beam_source = len(arrays_arg) > 0

    print("In build_pp_chache, after zeros")
    init_list = [
        {
            "charge": free_electrons_per_ion,
            "mass": free_electrons_per_ion,
            "is_ion": False,
        },
        {
            "charge": -free_electrons_per_ion,
            "mass": ion_mass / ct.m_e,
            "is_ion": True,
        },
    ]

    print("In build_pp_chache, before pp_initialize")
    species_list = pp_initialize(
        init_list,
        n_xi,
        ppc_n,
        dr,
        radial_density_normalized,
        ion_motion,
        False,   # critical: no history for cheap cache copies
        plasma_pusher,
    )
    pp_cache = tuple(s.serialize() for s in species_list)

    print("In build_pp_chache, before evolve_window")
    print(f"{bunch_source_arrays=}")
    print(f"{has_beam_source=}")

    # Evolve tail -> k_tail+1 so returned state is ready for slice k_tail.
    if has_beam_source:
        evolve_window_inplace_AG_beam(
            pp_cache,
            n_xi,
            n_r,
            dxi,
            dr,
            r_fld_n,
            has_laser_source,
            laser_a2,
            nabla_a2,
            arrays_arg,
            xi_idx_arg,
            meta_arg,
            max_gamma,
            ag_i_grid,
            ag_r_grid_n,
            ag_psi_grid,
            ag_bt_grid,
            p_shape,
            False,    # calculate_rho off
            rho,
            rho_e,
            rho_i,
            chi,
            False,    # no history
            ("none",),
            n_xi - 1,
            k_tail + 1,
        )
    else:
        evolve_window_inplace_AG_nobeam(
            pp_cache,
            n_xi,
            n_r,
            dxi,
            dr,
            r_fld_n,
            has_laser_source,
            laser_a2,
            nabla_a2,
            max_gamma,
            ag_i_grid,
            ag_r_grid_n,
            ag_psi_grid,
            ag_bt_grid,
            p_shape,
            False,    # calculate_rho off
            rho,
            rho_e,
            rho_i,
            chi,
            False,    # no history
            ("none",),
            n_xi - 1,
            k_tail + 1,
        )

    print("In build_pp_chache, after evolve_window")
    return pp_cache


def calculate_wakefields_ez_km1_from_cache(
    pp_state_kp1,            # cached plasma state ready for k
    k,                       # vary q_bunch at slice k, want Ez at k-1
    laser_a2,
    r_max,
    xi_min,
    xi_max,
    n_r,
    n_xi,
    n_p,
    radial_density,
    r_max_plasma,
    p_shape,
    max_gamma,
    bunch_source_arrays,
    bunch_source_xi_indices,
    bunch_source_metadata,
    fld_arrays,

    # single target AG
    ag_i_grid,
    ag_r_grid,      # physical units, no guards, includes border cells
    ag_psi_grid,
    ag_bt_grid,
):
    """
    Starting from cached plasma state at k+1, evolve only slices k..k-2
    and return Ez(k-1, r) on the target adaptive grid interior radial nodes.
    """
    print("In calculate_wakefields_ez_km1_from_cache, start")

    rho, rho_e, rho_i, chi, E_r, E_z, B_t, xi_fld, r_fld = fld_arrays

    s_d = ge.plasma_skin_depth(n_p * 1e-6)
    r_max_n = r_max / s_d
    xi_min_n = xi_min / s_d
    xi_max_n = xi_max / s_d
    dr = r_max_n / n_r
    dxi = (xi_max_n - xi_min_n) / (n_xi - 1)

    r_fld_n = r_fld / s_d
    ag_r_grid_n = ag_r_grid / s_d

    # Base-grid scratch arrays
    B_t[:] = 0.0
    chi[:] = 0.0
    rho[:] = 0.0
    rho_e[:] = 0.0
    rho_i[:] = 0.0
    E_r[:] = 0.0
    E_z[:] = 0.0

    # Target AG arrays for this trial
    ag_psi_grid[:] = 0.0
    ag_bt_grid[:] = 0.0

    has_laser_source = laser_a2 is not None
    if has_laser_source:
        n_xi_base = laser_a2.shape[0] - 4
        n_r_base = laser_a2.shape[1] - 4
        dr_base = r_fld_n[1] - r_fld_n[0]
        nabla_a2 = np.zeros((n_xi_base + 4, n_r_base + 4))
        radial_gradient(laser_a2[2:-2, 2:-2], dr_base, nabla_a2[2:-2, 2:-2])
    else:
        laser_a2 = np.zeros((0, 0))
        nabla_a2 = np.zeros((0, 0))

    arrays_arg, xi_idx_arg, meta_arg = _prepare_bunch_source_args(
        bunch_source_arrays,
        bunch_source_xi_indices,
        bunch_source_metadata,
    )
    has_beam_source = len(arrays_arg) > 0

    # Scratch state so cache itself is not mutated
    pp_trial = deepcopy_pp_state(pp_state_kp1)

    print("In calculate_wakefields_ez_km1_from_cache, before evolve_window")
    print(f"{has_beam_source=}")

    # Evolve k, k-1, k-2
    if has_beam_source:
        evolve_window_inplace_AG_beam(
            pp_trial,
            n_xi,
            n_r,
            dxi,
            dr,
            r_fld_n,
            has_laser_source,
            laser_a2,
            nabla_a2,
            arrays_arg,
            xi_idx_arg,
            meta_arg,
            max_gamma,
            ag_i_grid,
            ag_r_grid_n,
            ag_psi_grid,
            ag_bt_grid,
            p_shape,
            False,     # rho off
            rho,
            rho_e,
            rho_i,
            chi,
            False,     # no history
            ("none",),
            k,
            k - 2,
        )
    else:
        evolve_window_inplace_AG_nobeam(
            pp_trial,
            n_xi,
            n_r,
            dxi,
            dr,
            r_fld_n,
            has_laser_source,
            laser_a2,
            nabla_a2,
            max_gamma,
            ag_i_grid,
            ag_r_grid_n,
            ag_psi_grid,
            ag_bt_grid,
            p_shape,
            False,     # rho off
            rho,
            rho_e,
            rho_i,
            chi,
            False,     # no history
            ("none",),
            k,
            k - 2,
        )

    print(f"{k}")
    print(f"{ag_i_grid[0]}")
    print(f"{ag_i_grid[-1]}")
    # Compute Ez(k-1) using centered stencil from AG psi.
    # Need rows for k and k-2 to exist on the AG.
    if k < ag_i_grid[0] or k > ag_i_grid[-1]:
        raise ValueError("slice k is outside target adaptive-grid xi range")
    if (k - 2) < ag_i_grid[0] or (k - 2) > ag_i_grid[-1]:
        raise ValueError("slice k-2 is outside target adaptive-grid xi range")

    local_k = k + 2 - ag_i_grid[0]
    local_km2 = k - 2 + 2 - ag_i_grid[0]

    E_0 = ge.plasma_cold_non_relativisct_wave_breaking_field(n_p * 1e-6)

    # Interior radial nodes only, matching ag_r_grid
    psi_k = ag_psi_grid[local_k, 2:-2]
    psi_km2 = ag_psi_grid[local_km2, 2:-2]

    Ez_r_km1 = -(psi_k - psi_km2) / (2.0 * dxi) * E_0

    print("In calculate_wakefields_ez_km1_from_cache, after evolve_window")
    print("psi(k) axis/mean/min/max:", psi_k[0], psi_k.mean(), psi_k.min(), psi_k.max())
    print("psi(km2) axis/mean/min/max:", psi_km2[0], psi_km2.mean(), psi_km2.min(), psi_km2.max())

    return Ez_r_km1


def commit_cache_one_slice(
    pp_cache,
    k,
    laser_a2,
    r_max,
    xi_min,
    xi_max,
    n_r,
    n_xi,
    n_p,
    p_shape,
    max_gamma,
    bunch_source_arrays,
    bunch_source_xi_indices,
    bunch_source_metadata,
    fld_arrays,

    # single target AG
    ag_i_grid,
    ag_r_grid,      # physical units, no guards, includes border cells
    ag_psi_grid,
    ag_bt_grid,
):
    """
    Advance the plasma cache by exactly one slice k.
    In single-AG SALAME mode, only the target AG row(s) are written.
    """
    print("In commit_cache_one_slice, start")

    rho, rho_e, rho_i, chi, E_r, E_z, B_t, xi_fld, r_fld = fld_arrays

    s_d = ge.plasma_skin_depth(n_p * 1e-6)
    r_max_n = r_max / s_d
    xi_min_n = xi_min / s_d
    xi_max_n = xi_max / s_d
    dr = r_max_n / n_r
    dxi = (xi_max_n - xi_min_n) / (n_xi - 1)

    r_fld_n = r_fld / s_d
    ag_r_grid_n = ag_r_grid / s_d

    # Base-grid scratch arrays
    B_t[:] = 0.0
    chi[:] = 0.0
    rho[:] = 0.0
    rho_e[:] = 0.0
    rho_i[:] = 0.0
    E_r[:] = 0.0
    E_z[:] = 0.0

    # For safety, zero AG arrays too
    ag_psi_grid[:] = 0.0
    ag_bt_grid[:] = 0.0

    has_laser_source = laser_a2 is not None
    if has_laser_source:
        n_xi_base = laser_a2.shape[0] - 4
        n_r_base = laser_a2.shape[1] - 4
        dr_base = r_fld_n[1] - r_fld_n[0]
        nabla_a2 = np.zeros((n_xi_base + 4, n_r_base + 4))
        radial_gradient(laser_a2[2:-2, 2:-2], dr_base, nabla_a2[2:-2, 2:-2])
    else:
        laser_a2 = np.zeros((0, 0))
        nabla_a2 = np.zeros((0, 0))

    arrays_arg, xi_idx_arg, meta_arg = _prepare_bunch_source_args(
        bunch_source_arrays,
        bunch_source_xi_indices,
        bunch_source_metadata,
    )
    has_beam_source = len(arrays_arg) > 0

    print("In commit_cache_one_slice, before evolve_window")
    print(f"{has_beam_source=}")

    if has_beam_source:
        evolve_window_inplace_AG_beam(
            pp_cache,
            n_xi,
            n_r,
            dxi,
            dr,
            r_fld_n,
            has_laser_source,
            laser_a2,
            nabla_a2,
            arrays_arg,
            xi_idx_arg,
            meta_arg,
            max_gamma,
            ag_i_grid,
            ag_r_grid_n,
            ag_psi_grid,
            ag_bt_grid,
            p_shape,
            False,     # rho off
            rho,
            rho_e,
            rho_i,
            chi,
            False,     # no history
            ("none",),
            k,
            k,
        )
    else:
        evolve_window_inplace_AG_nobeam(
            pp_cache,
            n_xi,
            n_r,
            dxi,
            dr,
            r_fld_n,
            has_laser_source,
            laser_a2,
            nabla_a2,
            max_gamma,
            ag_i_grid,
            ag_r_grid_n,
            ag_psi_grid,
            ag_bt_grid,
            p_shape,
            False,     # rho off
            rho,
            rho_e,
            rho_i,
            chi,
            False,     # no history
            ("none",),
            k,
            k,
        )

    print("In commit_cache_one_slice, after evolve_window")
