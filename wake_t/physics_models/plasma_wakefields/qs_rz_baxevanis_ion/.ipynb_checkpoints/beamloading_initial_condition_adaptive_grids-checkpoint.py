# beamloading_initial_condition_adaptive_grids.py
from __future__ import annotations

import numpy as np
import scipy.constants as ct
import aptools.plasma_accel.general_equations as ge

from .solver_adaptive_grids import (
    build_pp_cache_at_kp1,
    calculate_wakefields_ez_km1_from_cache,
    commit_cache_one_slice,
)

from wake_t.particles.inverse_deposition import inverse_deposit_3d_distribution
from wake_t.particles.deposition import deposit_3d_distribution


def beamloading_initial_condition_adaptive_grids(
    *,
    # --- target bunch adaptive-grid geometry / arrays ---
    ag_i_grid: np.ndarray,
    ag_r_grid: np.ndarray,       # physical cells + border cells, no guards
    ag_xi_grid: np.ndarray,
    ag_dr: float,
    ag_dxi: float,
    ag_nr: int,                  # includes border cells, excludes guards
    ag_nxi: int,
    ag_r_max: float,
    ag_xi_min: float,
    ag_xi_max: float,
    ag_psi_grid: np.ndarray,     # (ag_nxi+4, ag_nr+4)
    ag_bt_grid: np.ndarray,      # usually witness_grid.b_t

    # --- base-grid arrays kept for solver scratch / laser / chi ---
    chi: np.ndarray,
    fld_arrays: list,

    # --- plasma / solver params from Quasistatic2DWakefieldIon ---
    n_p: float,
    ppc: np.ndarray,
    r_max_plasma: float,
    p_shape: str,
    max_gamma: float,
    plasma_pusher: str,
    ion_motion: bool,
    ion_mass: float,
    free_electrons_per_ion: int,

    # --- runtime inputs ---
    laser_a2,
    radial_density,

    # q_fixed from all non-target bunches, already expressed as bunch sources
    bunch_source_arrays: list,
    bunch_source_xi_indices: list,
    bunch_source_metadata: list,

    # target bunch
    bunch,
    q_var: np.ndarray,           # target bunch deposited on its own AG
):
    """
    SALAME initial-condition iteration on a single target adaptive grid.

    Inputs
    ------
    - bunch_source_arrays / xi_indices / metadata:
        fixed bunch-source contribution from all other bunches
    - q_var:
        target bunch deposited on its own AG (same shape as AG arrays: nxi+4,nr+4)
    - ag_* :
        target bunch adaptive-grid geometry/arrays

    Output / side effects
    ---------------------
    - updates q_var in-place logically through qb_var_current
    - updates bunch.w
    - writes final shaped deposit into q_var and ag_psi_grid/ag_bt_grid
    """

    # ----------------- helpers -----------------
    def q_line_from_q2d(q2d: np.ndarray) -> np.ndarray:
        q = q2d[2:-2, 2:-2]
        return np.sum(q, axis=1)

    def set_slice_line_charge_var(q_var2d: np.ndarray, k: int, g_new: float) -> np.ndarray:
        q_new = q_var2d.copy()
        g_old_all = q_line_from_q2d(q_var2d)
        g_old = g_old_all[k]
        if g_old == 0.0:
            return q_new
        q_new[2 + k, 2:-2] *= (g_new / g_old)
        return q_new

    def Ez_weighted_at_slice(Ez_r: np.ndarray, q_var2d: np.ndarray, slice_i: int) -> float:
        q_slice = q_var2d[2 + slice_i, 2:-2]
        w = np.abs(q_slice)
        s = w.sum()
        if s == 0.0:
            w = np.ones_like(q_slice)
            s = w.sum()
        return float((Ez_r * w).sum() / s)

    def solve_Ez_weighted_km1_cached(q_var2d: np.ndarray, pp_cache, km1: int) -> float:
        """
        For current variable target bunch deposit q_var2d, compute weighted Ez at slice km1
        using cached plasma state ready for slice k = km1+1.
        """
        k = km1 + 1

        Ez_r = calculate_wakefields_ez_km1_from_cache(
            pp_state_kp1=pp_cache,
            k=k,
            laser_a2=laser_a2,
            r_max=ag_r_max,
            xi_min=ag_xi_min,
            xi_max=ag_xi_max,
            n_r=ag_nr,
            n_xi=ag_nxi,
            n_p=n_p,
            radial_density=radial_density,
            r_max_plasma=r_max_plasma,
            p_shape=p_shape,
            max_gamma=max_gamma,
            bunch_source_arrays=bunch_source_arrays + [q_var_to_bt(q_var2d)],
            bunch_source_xi_indices=bunch_source_xi_indices + [ag_i_grid],
            bunch_source_metadata=bunch_source_metadata + [ag_metadata],
            fld_arrays=fld_arrays,
            ag_i_grid=ag_i_grid,
            ag_r_grid=ag_r_grid,
            ag_psi_grid=ag_psi_grid,
            ag_bt_grid=ag_bt_grid,
        )
        return Ez_weighted_at_slice(Ez_r, q_var2d, km1)

    def q_var_to_bt(q_var2d: np.ndarray) -> np.ndarray:
        """
        Convert target AG q_var to bunch-source b_theta array on the same AG.
        """
        bt = np.zeros_like(q_var2d)
        from .b_theta_bunch import calculate_bunch_source
        calculate_bunch_source(q_var2d, ag_nr, ag_nxi, bt)
        return bt


    print("start SALAME solver")
    # ----------------- sanity / setup -----------------
    if len(bunch_source_arrays) != len(bunch_source_xi_indices) or len(bunch_source_arrays) != len(bunch_source_metadata):
        raise ValueError("bunch_source_arrays / xi_indices / metadata must have same length")

    s_d = ge.plasma_skin_depth(n_p * 1e-6)
    ag_metadata = np.array(
        [
            ag_r_grid[0],
            ag_r_grid[-1] + 2 * ag_dr,   # mimic base-grid metadata convention
            ag_dr,
        ]
    ) / s_d

    qb_var_current = q_var.copy()

    g_line_var0 = q_line_from_q2d(qb_var_current)
    support = np.where(np.abs(g_line_var0) > 0.0)[0]
    if support.size < 2:
        # nothing to shape, but still map current AG deposit back to bunch weights
        qb_var_final = qb_var_current
    else:
        k_tail = support[-1]

        print("before build_pp_cache")
        # Build cache ready for slice (k_tail + 1), with only fixed bunch sources.
        pp_cache = build_pp_cache_at_kp1(
            k_tail=k_tail + 1,
            laser_a2=laser_a2,
            r_max=ag_r_max,
            xi_min=ag_xi_min,
            xi_max=ag_xi_max,
            n_r=ag_nr,
            n_xi=ag_nxi,
            ppc=ppc,
            n_p=n_p,
            r_max_plasma=r_max_plasma,
            radial_density=radial_density,
            p_shape=p_shape,
            max_gamma=max_gamma,
            plasma_pusher=plasma_pusher,
            ion_motion=ion_motion,
            ion_mass=ion_mass,
            free_electrons_per_ion=free_electrons_per_ion,
            bunch_source_arrays=bunch_source_arrays,
            bunch_source_xi_indices=bunch_source_xi_indices,
            bunch_source_metadata=bunch_source_metadata,
            fld_arrays=fld_arrays,
            ag_i_grid=ag_i_grid,
            ag_r_grid=ag_r_grid,
            ag_psi_grid=ag_psi_grid,
            ag_bt_grid=ag_bt_grid,
        )
        print("after build_pp_cache")

        # Tail target field: fixed bunches + current target bunch
        Ez_target = solve_Ez_weighted_km1_cached(qb_var_current, pp_cache, k_tail)
        print(f"{Ez_target=}")

        # Advance cache by one slice so iteration can proceed from k_tail downward
        # using only fixed bunches. Variable bunch slice k is added only in the trial solve.
        commit_cache_one_slice(
            pp_cache=pp_cache,
            k=k_tail + 1,
            laser_a2=laser_a2,
            r_max=ag_r_max,
            xi_min=ag_xi_min,
            xi_max=ag_xi_max,
            n_r=ag_nr,
            n_xi=ag_nxi,
            n_p=n_p,
            p_shape=p_shape,
            max_gamma=max_gamma,
            bunch_source_arrays=bunch_source_arrays,
            bunch_source_xi_indices=bunch_source_xi_indices,
            bunch_source_metadata=bunch_source_metadata,
            fld_arrays=fld_arrays,
            ag_i_grid=ag_i_grid,
            ag_r_grid=ag_r_grid,
            ag_psi_grid=ag_psi_grid,
            ag_bt_grid=ag_bt_grid,
        )

        max_iter = 100
        tol = 1e-4

        for k in range(k_tail, support[0], -1):
            print(k)
            if np.abs(g_line_var0[k]) == 0.0:
                continue

            g_min = -1e-100
            g_max = 5.0 * g_line_var0[k]

            qb_min = set_slice_line_charge_var(qb_var_current, k, g_min)
            Ez_min_km1 = solve_Ez_weighted_km1_cached(qb_min, pp_cache, k - 1)

            qb_max = set_slice_line_charge_var(qb_var_current, k, g_max)
            Ez_max_km1 = solve_Ez_weighted_km1_cached(qb_max, pp_cache, k - 1)

            print("g_old =", q_line_from_q2d(qb_var_current)[k])
            print("g_min slice =", q_line_from_q2d(qb_min)[k])
            print("g_max slice =", q_line_from_q2d(qb_max)[k])
            print(f"{Ez_max_km1=}")
            print(f"{Ez_min_km1=}")

            while np.abs(Ez_max_km1) > np.abs(Ez_target):
                g_max *= 5.0
                qb_max = set_slice_line_charge_var(qb_var_current, k, g_max)
                Ez_max_km1 = solve_Ez_weighted_km1_cached(qb_max, pp_cache, k - 1)
                print(f"{g_max=}")
                print(f"{Ez_max_km1=}")
                if g_max == 0.0 or not np.isfinite(g_max):
                    break

            qb_new = qb_var_current
            Ez_new_km1 = Ez_min_km1

            for _ in range(max_iter):
                den = np.abs(Ez_max_km1 - Ez_min_km1)
                if den == 0.0:
                    break

                if Ez_target < Ez_min_km1:
                    g_new = g_min
                    qb_try = set_slice_line_charge_var(qb_var_current, k, g_new)
                    Ez_try_km1 = solve_Ez_weighted_km1_cached(qb_try, pp_cache, k - 1)
                    qb_new, Ez_new_km1 = qb_try, Ez_try_km1
                    print(f"{Ez_try_km1=}")
                    print("Need positrons for this slice...")
                    break

                wg = np.abs(Ez_target - Ez_min_km1) / den
                g_new = wg * g_max + (1.0 - wg) * g_min

                qb_try = set_slice_line_charge_var(qb_var_current, k, g_new)
                Ez_try_km1 = solve_Ez_weighted_km1_cached(qb_try, pp_cache, k - 1)

                print(f"{Ez_try_km1=}")

                if np.abs(Ez_try_km1) > np.abs(Ez_target):
                    g_min, Ez_min_km1 = g_new, Ez_try_km1
                else:
                    g_max, Ez_max_km1 = g_new, Ez_try_km1

                rel = np.abs(Ez_try_km1 - Ez_target) / (np.abs(Ez_target) + 1e-300)
                qb_new, Ez_new_km1 = qb_try, Ez_try_km1
                print(f"{rel=}")
                if rel < tol:
                    break

            qb_var_current = qb_new

            # commit only fixed sources to cache at slice k
            # variable bunch remains trial-only in SALAME root search
            commit_cache_one_slice(
                pp_cache=pp_cache,
                k=k,
                laser_a2=laser_a2,
                r_max=ag_r_max,
                xi_min=ag_xi_min,
                xi_max=ag_xi_max,
                n_r=ag_nr,
                n_xi=ag_nxi,
                n_p=n_p,
                p_shape=p_shape,
                max_gamma=max_gamma,
                bunch_source_arrays=bunch_source_arrays,
                bunch_source_xi_indices=bunch_source_xi_indices,
                bunch_source_metadata=bunch_source_metadata,
                fld_arrays=fld_arrays,
                ag_i_grid=ag_i_grid,
                ag_r_grid=ag_r_grid,
                ag_psi_grid=ag_psi_grid,
                ag_bt_grid=ag_bt_grid,
            )

        qb_var_final = qb_var_current

    # ----------------- write final target AG deposit -----------------
    q_var[:, :] = qb_var_final

    # sanitize chi to avoid NaN/Inf killing laser envelope
    chi_int = chi[2:-2, 2:-2]
    if not np.all(np.isfinite(chi_int)):
        chi_int[~np.isfinite(chi_int)] = 0.0

    # ----------------- inverse map AG grid deposit -> target bunch particle weights -----------------
    q_gathered, _ = inverse_deposit_3d_distribution(
        bunch.xi,
        bunch.x,
        bunch.y,
        ag_xi_grid[0],
        ag_r_grid[0],
        ag_nxi,
        ag_nr,
        ag_dxi,
        ag_dr,
        q_var,
        p_shape=p_shape,
        use_ruyten=True,
        r_min_deposit=0.0,
    )

    count_grid = np.zeros_like(q_var)
    ones = np.ones_like(bunch.xi)

    deposit_3d_distribution(
        bunch.xi,
        bunch.x,
        bunch.y,
        ones,
        ag_xi_grid[0],
        ag_r_grid[0],
        ag_nxi,
        ag_nr,
        ag_dxi,
        ag_dr,
        count_grid,
        p_shape=p_shape,
        use_ruyten=True,
        r_min_deposit=0.0,
    )

    count_p, _ = inverse_deposit_3d_distribution(
        bunch.xi,
        bunch.x,
        bunch.y,
        ag_xi_grid[0],
        ag_r_grid[0],
        ag_nxi,
        ag_nr,
        ag_dxi,
        ag_dr,
        count_grid,
        p_shape=p_shape,
        use_ruyten=True,
        r_min_deposit=0.0,
    )

    eps = 1e-30
    q_inv = q_gathered / np.maximum(count_p, eps)

    rho1 = np.zeros_like(q_var)
    deposit_3d_distribution(
        bunch.xi,
        bunch.x,
        bunch.y,
        q_inv,
        ag_xi_grid[0],
        ag_r_grid[0],
        ag_nxi,
        ag_nr,
        ag_dxi,
        ag_dr,
        rho1,
        p_shape=p_shape,
        use_ruyten=True,
        r_min_deposit=0.0,
    )
    absdiff = np.max(np.abs(rho1 - q_var))
    reldiff = absdiff / (np.max(np.abs(q_var)) + 1e-30)
    print("inverse redeposit max abs diff:", absdiff)
    print("inverse redeposit rel diff    :", reldiff)
    print("sum grid:", np.sum(q_var), "sum inv:", np.sum(q_inv))

    s_d_phys = ct.c / np.sqrt(ct.e**2 * n_p / (ct.m_e * ct.epsilon_0))
    k_norm = 1.0 / (2 * np.pi * ct.e * ag_dr * ag_dxi * s_d_phys * n_p)
    q_normalized = q_inv / k_norm

    bunch.w = q_normalized / bunch.q_species
