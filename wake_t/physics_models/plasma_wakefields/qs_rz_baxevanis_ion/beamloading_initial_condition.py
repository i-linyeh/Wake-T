# beamloading_initial_condition.py
from __future__ import annotations

import numpy as np
import scipy.constants as ct
import aptools.plasma_accel.general_equations as ge

from .b_theta_bunch import calculate_bunch_source, calculate_bunch_source_slice
from .solver import (
    calculate_wakefields,
    calculate_wakefields_ez_slice,
    calculate_wakefields_ez_km1_from_cache,
    build_pp_cache_at_kp1,
    commit_cache_one_slice,
)

from wake_t.particles.inverse_deposition import inverse_deposit_3d_distribution, inverse_deposit_fast, inverse_deposit_lsqr
from wake_t.particles.deposition import deposit_3d_distribution


def beamloading_initial_condition(
    *,
    # --- grid arrays / geometry (base grid OR adaptive grid) ---
    q_bunch: np.ndarray,
    b_t_bunch: np.ndarray,
    chi: np.ndarray,
    r_fld: np.ndarray,
    xi_fld: np.ndarray,
    dr: float,
    dxi: float,
    n_r: int,
    n_xi: int,

    # --- solver / plasma parameters (always from Quasistatic2DWakefieldIon) ---
    n_p: float,
    ppc: np.ndarray,
    r_max: float,
    xi_min: float,
    xi_max: float,
    r_max_plasma: float,
    p_shape: str,
    max_gamma: float,
    plasma_pusher: str,
    ion_motion: bool,
    ion_mass: float,
    free_electrons_per_ion: int,
    field_diags: list,
    fld_arrays: list,

    # --- runtime inputs ---
    laser_a2,
    radial_density,
    bunch_source_arrays: list,
    bunch_source_xi_indices: list,
    bunch_source_metadata: list,
    bunch,  # single ParticleBunch

    q_fixed: np.ndarray,
    q_var: np.ndarray,


    ):
    """
    Standalone SALAME / beam-loading initial condition step.

    Keeps all your prints inside this function, so both base-grid and adaptive-grid
    codepaths behave the same.

    NOTE:
      - This function mutates q_bunch, b_t_bunch, chi and updates bunch.w.
      - It assumes q_bunch/b_t_bunch include guard cells (same convention as your code).
    """

    # --- keep your existing bunch_source init ---
    if len(bunch_source_arrays) == 0:
        bunch_source_arrays.append(b_t_bunch)
        bunch_source_xi_indices.append(np.arange(n_xi))

        s_d = ge.plasma_skin_depth(n_p * 1e-6)
        bunch_source_metadata.append(
            np.array([r_fld[0], r_fld[-1] + 2 * dr, dr]) / s_d
        )

    # ----------------- helpers (local) -----------------
    r_centers = r_fld
    xi_centers = xi_fld

    def q_bunch_line_from_qbunch(qb2d: np.ndarray) -> np.ndarray:
        qb = qb2d[2:-2, 2:-2]  # (n_xi,n_r)
        return np.sum(qb, axis=1)

    def Ez_weighted_at_slice(Ez_r: np.ndarray, qb2d: np.ndarray, slice_i: int) -> float:
        qb_slice = qb2d[2 + slice_i, 2:-2]
        w = np.abs(qb_slice)
        s = w.sum()
        if s == 0.0:
            w = np.ones(len(w))
            s = w.sum()
        return float((Ez_r[2:-2] * w).sum() / s)

#    def set_slice_line_charge(qb2d: np.ndarray, k: int, g_new: float) -> np.ndarray:
#        qb_new = qb2d.copy()
#        g_old_all = q_bunch_line_from_qbunch(qb2d)
#        g_old = g_old_all[k]
#        if g_old == 0.0:
#            return qb_new
#        s = g_new / g_old
#        qb_new[2 + k, 2:-2] *= s
#        return qb_new



    def set_slice_line_charge_var(qb_var2d: np.ndarray, k: int, g_new: float) -> np.ndarray:
        qb_new = qb_var2d.copy()
        g_old_all = q_bunch_line_from_qbunch(qb_var2d)
        g_old = g_old_all[k]
        if g_old == 0.0:
            return qb_new
        s = g_new / g_old
        qb_new[2 + k, 2:-2] *= s
        return qb_new




    
    def Ez_weighted_from_fields(Ez2d: np.ndarray, qb2d: np.ndarray) -> np.ndarray:
        Ez = Ez2d[2:-2, 2:-2]          # (n_xi,n_r)
        qb = qb2d[2:-2, 2:-2]
        w = np.abs(qb)
        wsum = np.sum(w, axis=1)
        out = np.zeros(Ez.shape[0], dtype=float)
        m = wsum > 0
        out[m] = np.sum(Ez[m, :] * w[m, :], axis=1) / wsum[m]
        return out
    
    
    def solve_with_qbunch(qb2d: np.ndarray):
        """
        Full wake solve with the current qbunch (same as the method version).
        Updates fld_arrays in-place via calculate_wakefields(...).
        Returns:
          Ez_w : bunch-weighted <Ez>(xi)
          g_line: line-charge proxy from q_bunch
          pp    : plasma particle history/state returned by calculate_wakefields
        """
        # update q_bunch and b_t_bunch
        q_bunch[:, :] = qb2d
        calculate_bunch_source(q_bunch, n_r, n_xi, b_t_bunch)
    
        # overwrite base-grid slot (index 0)
        bunch_source_arrays[0] = b_t_bunch
    
        calculate_rho = any("rho" in diag for diag in field_diags)
    
        pp = calculate_wakefields(
            laser_a2,
            r_max,
            xi_min,
            xi_max,
            n_r,
            n_xi,
            ppc,
            n_p,
            r_max_plasma=r_max_plasma,
            radial_density=radial_density,
            p_shape=p_shape,
            max_gamma=max_gamma,
            plasma_pusher=plasma_pusher,
            ion_motion=ion_motion,
            ion_mass=ion_mass,
            free_electrons_per_ion=free_electrons_per_ion,
            fld_arrays=fld_arrays,
            bunch_source_arrays=bunch_source_arrays,
            bunch_source_xi_indices=bunch_source_xi_indices,
            bunch_source_metadata=bunch_source_metadata,
            store_plasma_history=False,      # IC stage only
            calculate_rho=calculate_rho,
            particle_diags=[],
        )
    
        # fld_arrays[5] is e_z in your base-grid layout
        e_z_arr = fld_arrays[5]
        Ez_w = Ez_weighted_from_fields(e_z_arr, q_bunch)
        g_line = q_bunch_line_from_qbunch(q_bunch)
        return Ez_w, g_line, pp
    
    



#    def solve_Ez_weighted_km1_cached(qb2d: np.ndarray, pp_cache, km1: int) -> float:
#        k = km1 + 1
#        if k < 2:
#            return 0.0
#
#        q_bunch[:, :] = qb2d
#        calculate_bunch_source_slice(q_bunch, n_r, k, b_t_bunch)
#        bunch_source_arrays[0] = b_t_bunch
#
#        Ez_r = calculate_wakefields_ez_km1_from_cache(
#            pp_cache, k,
#            laser_a2, r_max, xi_min, xi_max,
#            n_r, n_xi, n_p,
#            radial_density=radial_density,
#            r_max_plasma=r_max_plasma,
#            p_shape=p_shape,
#            max_gamma=max_gamma,
#            bunch_source_arrays=bunch_source_arrays,
#            bunch_source_xi_indices=bunch_source_xi_indices,
#            bunch_source_metadata=bunch_source_metadata,
#            fld_arrays=fld_arrays,
#        )
#        return Ez_weighted_at_slice(Ez_r, q_bunch, km1)



    def solve_Ez_weighted_km1_cached(qb_var2d: np.ndarray, pp_cache, km1: int) -> float:
        qb_total = qb_total_from_var(qb_var2d)
    
        q_bunch[:, :] = qb_total
        k = km1 + 1
        calculate_bunch_source_slice(q_bunch, n_r, k, b_t_bunch)
        bunch_source_arrays[0] = b_t_bunch
    
        Ez_r = calculate_wakefields_ez_km1_from_cache(
            pp_cache, k,
            laser_a2, r_max, xi_min, xi_max,
            n_r, n_xi, n_p,
            radial_density=radial_density,
            r_max_plasma=r_max_plasma,
            p_shape=p_shape,
            max_gamma=max_gamma,
            bunch_source_arrays=bunch_source_arrays,
            bunch_source_xi_indices=bunch_source_xi_indices,
            bunch_source_metadata=bunch_source_metadata,
            fld_arrays=fld_arrays,
        )
        # weight using witness slice only:
        return Ez_weighted_at_slice(Ez_r, qb_var2d, km1)






    def qb_total_from_var(qb_var: np.ndarray) -> np.ndarray:
        return qb_fixed + qb_var










    # ----------------- SALAME iteration -----------------

    #qb_current = q_bunch.copy()

    #g_line0 = q_bunch_line_from_qbunch(qb_current)
    #support = np.where(np.abs(g_line0) > 0.0)[0]
    #if support.size < 2:
    #    return

    #k_tail = support[-1]


    qb_current = q_bunch.copy()


    qb_fixed = q_fixed.copy()
    qb_var_current = q_var.copy()
    


    
    g_line_var0 = q_bunch_line_from_qbunch(qb_var_current)
    support = np.where(np.abs(g_line_var0) > 0.0)[0]
    k_tail = support[-1]



    Ez_w0, g_line0, pp0 = solve_with_qbunch(qb_current)
    
    # define bunch-support indices (where there is charge)
    #support = np.where(np.abs(g_line0) > 0.0)[0]
    #if support.size < 2:
    #    # nothing to shape
    #    return
    
    #k_tail = support[-1]
    
    Ez_target = Ez_w0[k_tail]   # tail value target (flat)
    print(f"{Ez_target=}") 

    # ------------------------------------------------------------
    # Build plasma cache: state AFTER slice (k_tail + 1)
    # ------------------------------------------------------------
    pp_cache = build_pp_cache_at_kp1(
        k_tail,
        laser_a2,
        r_max,
        xi_min,
        xi_max,
        n_r,
        n_xi,
        ppc,
        n_p,
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
    )

    # Get Ez_target at k_tail (control point is k_tail-1, but your code uses tail value)
    # We'll compute Ez at km1=k_tail-1 using current qbunch as a reference.
    #Ez_target = solve_Ez_weighted_km1_cached(qb_current, pp_cache, k_tail - 1)
    #print(f"{Ez_target=}")

    max_iter = 10
    tol = 1e-4

    for k in range(k_tail, support[0], -1):
        print(k)
        if np.abs(g_line0[k]) == 0.0:
            continue

        g_min = -1e-100
        g_max = 5.0 * g_line0[k]
        print(f"{g_max=}")

        qb_min = set_slice_line_charge_var(qb_var_current, k, g_min)
        Ez_min_km1 = solve_Ez_weighted_km1_cached(qb_min, pp_cache, k - 1)

        qb_max = set_slice_line_charge_var(qb_var_current, k, g_max)
        Ez_max_km1 = solve_Ez_weighted_km1_cached(qb_max, pp_cache, k - 1)

        print("g_old =", q_bunch_line_from_qbunch(qb_var_current)[k])
        print("g_min slice =", q_bunch_line_from_qbunch(qb_min)[k])
        print("g_max slice =", q_bunch_line_from_qbunch(qb_max)[k])
        print(f"{Ez_max_km1=}")
        print(f"{Ez_min_km1=}")

        while np.abs(Ez_max_km1) > np.abs(Ez_target):
            print(f"{g_max=}")
            g_max *= 5.0
            qb_max = set_slice_line_charge_var(qb_var_current, k, g_max)
            Ez_max_km1 = solve_Ez_weighted_km1_cached(qb_max, pp_cache, k - 1)
            print(f"{Ez_max_km1=}")
            if g_max == 0.0 or not np.isfinite(g_max):
                break

        qb_new = qb_var_current
        Ez_new_km1 = Ez_min_km1
        for _ in range(max_iter):
            den = np.abs(Ez_max_km1 - Ez_min_km1)
            if den == 0.0:
                break

            print(f"{Ez_max_km1=}")
            print(f"{Ez_min_km1=}")
            print(f"{Ez_target=}")

            if Ez_target < Ez_min_km1:
                g_new = g_min
                qb_try = set_slice_line_charge_var(qb_var_current, k, g_new)
                Ez_try_km1 = solve_Ez_weighted_km1_cached(qb_try, pp_cache, k - 1)
                qb_new, Ez_new_km1 = qb_try, Ez_try_km1
                print(f"{Ez_try_km1=}")
                print("Need positrons for this slice...")
                break

            wg = np.abs(Ez_target - Ez_min_km1) / den
            print(f"{wg=}")
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

        q_bunch[:, :] = qb_total_from_var(qb_var_current) 

        commit_cache_one_slice(
            pp_cache,
            k,
            laser_a2,
            r_max,
            xi_min,
            xi_max,
            n_r,
            n_xi,
            n_p,
            p_shape=p_shape,
            max_gamma=max_gamma,
            bunch_source_arrays=bunch_source_arrays,
            bunch_source_xi_indices=bunch_source_xi_indices,
            bunch_source_metadata=bunch_source_metadata,
            fld_arrays=fld_arrays,
        )

    #q_bunch[:, :] = qb_current 
    qb_total_final = qb_total_from_var(qb_var_current)
    q_bunch[:, :] = qb_total_final





    np.savez('q_bunch_current.npz', q_bunch=q_bunch) 




    # sanitize chi to avoid NaNs/Infs killing laser envelope 
    chi_int = chi[2:-2, 2:-2]
    if not np.all(np.isfinite(chi_int)):
        chi_int[~np.isfinite(chi_int)] = 0.0

    # map shaped deposited qbunch -> particle charges, update bunch weights
#    s_d = ge.plasma_skin_depth(n_p * 1e-6)

#    q_new, _ = inverse_deposit_3d_distribution(
#        bunch.xi, bunch.x, bunch.y,
#        xi_fld[0], r_fld[0],
#        n_xi, n_r, dxi, dr,
#        qb_var_current,
#        p_shape=p_shape,
#        use_ruyten=True,
#    )
#
#    count_grid = np.zeros_like(q_bunch)
#    ones = np.ones_like(bunch.q)
#
#    deposit_3d_distribution(
#        bunch.xi, bunch.x, bunch.y,
#        ones,
#        xi_fld[0],
#        r_fld[0],
#        n_xi,
#        n_r,
#        dxi,
#        dr,
#        count_grid,
#        p_shape=p_shape,
#        use_ruyten=True,
#    )
#
#    count_p, _ = inverse_deposit_3d_distribution(
#        bunch.xi, bunch.x, bunch.y,
#        xi_fld[0], r_fld[0],
#        n_xi, n_r, dxi, dr,
#        count_grid,
#        p_shape=p_shape,
#        use_ruyten=True,
#    )

    print(np.sum(qb_var_current))


    #print(qb_var_current)

    #dep = np.zeros((n_xi + 4, n_r + 4))
    #dep[2:-2, 2:-2] = qb_var_current


    qb_var_target = np.zeros_like(qb_var_current)
    qb_var_target[2:-2, 2:-2] = qb_var_current[2:-2, 2:-2]

    print(np.sum(qb_var_target))



    count_grid = np.zeros_like(qb_var_current)
    ones = np.ones_like(bunch.xi)
    
    deposit_3d_distribution(
        bunch.xi, bunch.x, bunch.y,
        ones,
        xi_fld[0], r_fld[0],
        n_xi, n_r, dxi, dr,
        count_grid,
        p_shape="cubic",
        use_ruyten=True,
        r_min_deposit=0.0,
    )
    
    mask = count_grid > 0
    print("sum qb_var_current total      =", np.sum(qb_var_current))
    print("sum qb_var_current where mask =", np.sum(qb_var_current[mask]))
    print("sum qb_var_current off-support =", np.sum(qb_var_current[~mask]))
    print("active support frac (grid)    =", np.mean(mask))




#    q_inv, info = inverse_deposit_fast(
#        bunch.xi, bunch.x, bunch.y, xi_fld[0], r_fld[0], n_xi, n_r, dxi, dr, qb_var_target,
#        use_ruyten=True,
#        r_min_deposit=0.0,
#        sign_mode="preserve_sign_of_init",   # or "preserve_sign_of_init"
#        n_iter=500,
#        alpha=0.05,
#    )
#
#
#    print(np.sum(q_inv))
#
#
#
#    rho1 = np.zeros_like(qb_var_current)
#    deposit_3d_distribution(
#        bunch.xi, bunch.x, bunch.y,
#        q_inv,
#        xi_fld[0], r_fld[0],
#        n_xi, n_r,
#        dxi, dr,
#        rho1,
#        p_shape="cubic",
#        use_ruyten=True,
#        r_min_deposit=0.0,
#    )
#    
#    absdiff = np.max(np.abs(rho1 - qb_var_current))
#    reldiff = absdiff / (np.max(np.abs(qb_var_current)) + 1e-30)
#    print("inverse redeposit max abs diff:", absdiff)
#    print("inverse redeposit rel diff    :", reldiff)
#    print("sum grid:", np.sum(qb_var_current), "sum inv:", np.sum(q_inv))







    # 1) Solve inverse by LSQR (masked)
#    fixed_sign = -1  # your qb_var_current sum is negative (electrons)
#    q_inv, info, count_grid, mask = inverse_deposit_lsqr(
#        bunch.xi, bunch.x, bunch.y,
#        xi_fld[0], r_fld[0], n_xi, n_r, dxi, dr,
#        qb_var_current,
#        use_ruyten=True,
#        r_min_deposit=0.0,
#        damp=1e-14,       # small damping helps conditioning; try 0, 1e-14, 1e-12
#        atol=1e-12,
#        btol=1e-12,
#        iter_lim=1000,
#        enforce_total=True,
#        fixed_sign=fixed_sign,
#    )





    # Simple normalized gather (correct pseudoinverse for dense particle coverage)
    q_gathered, _ = inverse_deposit_3d_distribution(
        bunch.xi, bunch.x, bunch.y,
        xi_fld[0], r_fld[0],
        n_xi, n_r, dxi, dr,
        qb_var_current,
        p_shape=p_shape,
        use_ruyten=True,
        r_min_deposit=0.0,
    )
    
    count_grid = np.zeros_like(qb_var_current)
    ones = np.ones_like(bunch.xi)
    deposit_3d_distribution(
        bunch.xi, bunch.x, bunch.y, ones,
        xi_fld[0], r_fld[0], n_xi, n_r, dxi, dr,
        count_grid,
        p_shape=p_shape,
        use_ruyten=True,
        r_min_deposit=0.0,
    )
    
    # Gather the count grid back to particles
    count_p, _ = inverse_deposit_3d_distribution(
        bunch.xi, bunch.x, bunch.y,
        xi_fld[0], r_fld[0],
        n_xi, n_r, dxi, dr,
        count_grid,
        p_shape=p_shape,
        use_ruyten=True,
        r_min_deposit=0.0,
    )
    
    eps = 1e-30
    q_inv = q_gathered / np.maximum(count_p, eps)
    
    # Enforce total charge exactly
    target_sum = np.sum(qb_var_current)
    inv_sum = np.sum(q_inv)
    if inv_sum != 0.0:
        q_inv *= (target_sum / inv_sum)







#    # For each particle, its weight = (slice line charge) / (number of particles in that slice)
#    # This is exact by construction
#    
#    q_inv = np.zeros_like(bunch.xi)
#    
#    for k_slice in range(n_xi):
#        # find particles in this xi slice
#        xi_lo = xi_fld[k_slice] - dxi/2
#        xi_hi = xi_fld[k_slice] + dxi/2
#        mask_p = (bunch.xi >= xi_lo) & (bunch.xi < xi_hi)
#        n_in_slice = np.sum(mask_p)
#        if n_in_slice == 0:
#            continue
#        
#        # total normalized charge in this slice from the shaped grid
#        g_slice = np.sum(qb_var_current[2 + k_slice, 2:-2])
#        
#        # distribute equally among particles in this slice
#        q_inv[mask_p] = g_slice / n_in_slice








#    # Step 1: normalized gather as initial guess (your current method)
#    q_gathered, _ = inverse_deposit_3d_distribution(
#        bunch.xi, bunch.x, bunch.y,
#        xi_fld[0], r_fld[0],
#        n_xi, n_r, dxi, dr,
#        qb_var_current,
#        p_shape=p_shape,
#        use_ruyten=True,
#        r_min_deposit=0.0,
#    )
#    
#    count_grid = np.zeros_like(qb_var_current)
#    ones = np.ones_like(bunch.xi)
#    deposit_3d_distribution(
#        bunch.xi, bunch.x, bunch.y, ones,
#        xi_fld[0], r_fld[0], n_xi, n_r, dxi, dr,
#        count_grid,
#        p_shape=p_shape,
#        use_ruyten=True,
#        r_min_deposit=0.0,
#    )
#    
#    count_p, _ = inverse_deposit_3d_distribution(
#        bunch.xi, bunch.x, bunch.y,
#        xi_fld[0], r_fld[0],
#        n_xi, n_r, dxi, dr,
#        count_grid,
#        p_shape=p_shape,
#        use_ruyten=True,
#        r_min_deposit=0.0,
#    )
#    
#    eps = 1e-30
#    q_inv = q_gathered / np.maximum(count_p, eps)
#    
#    # Step 2: iterative refinement
#    n_iter = 200
#    alpha = 0.5  # conservative step; tune if needed
#    
#    rho_buf = np.zeros_like(qb_var_current)
#    for it in range(n_iter):
#        rho_buf.fill(0.0)
#        deposit_3d_distribution(
#            bunch.xi, bunch.x, bunch.y, q_inv,
#            xi_fld[0], r_fld[0], n_xi, n_r, dxi, dr,
#            rho_buf,
#            p_shape=p_shape,
#            use_ruyten=True,
#            r_min_deposit=0.0,
#        )
#    
#        residual = qb_var_current - rho_buf   # grid residual
#    
#        # print every 20 iters to monitor convergence
#        if it % 20 == 0:
#            reldiff = np.max(np.abs(residual)) / (np.max(np.abs(qb_var_current)) + eps)
#            print(f"  iter {it:4d}  rel_diff={reldiff:.6e}  sum_inv={np.sum(q_inv):.6e}")
#    
#        # gather residual back to particles, normalized by count_p
#        res_gathered, _ = inverse_deposit_3d_distribution(
#            bunch.xi, bunch.x, bunch.y,
#            xi_fld[0], r_fld[0],
#            n_xi, n_r, dxi, dr,
#            residual,
#            p_shape=p_shape,
#            use_ruyten=True,
#            r_min_deposit=0.0,
#        )
#        
#        dw = res_gathered / np.maximum(count_p, eps)
#        q_inv = q_inv + alpha * dw
#    
#    # Step 3: enforce total charge exactly
#    target_sum = np.sum(qb_var_current)
#    inv_sum = np.sum(q_inv)
#    if inv_sum != 0.0:
#        q_inv *= (target_sum / inv_sum)
#    
#    print(f"Final: sum_grid={target_sum:.6e}  sum_inv={np.sum(q_inv):.6e}")





    # 2) Check redeposit diff
    rho1 = np.zeros_like(qb_var_current)
    deposit_3d_distribution(
        bunch.xi, bunch.x, bunch.y,
        q_inv,
        xi_fld[0], r_fld[0],
        n_xi, n_r, dxi, dr,
        rho1,
        p_shape="cubic",
        use_ruyten=True,
        r_min_deposit=0.0,
    )
    absdiff = np.max(np.abs(rho1 - qb_var_current))
    reldiff = absdiff / (np.max(np.abs(qb_var_current)) + 1e-30)
    #print("LSQR info:", info)
    print("inverse redeposit max abs diff:", absdiff)
    print("inverse redeposit rel diff    :", reldiff)
    print("sum grid:", np.sum(qb_var_current), "sum inv:", np.sum(q_inv))










#
#    target_sum = np.sum(qb_var_target)   # or qb_var_target.sum() (guards are zero)
#    inv_sum = np.sum(q_inv)
#    if inv_sum != 0.0:
#        q_inv *= (target_sum / inv_sum)
#

    s_d = ct.c / np.sqrt(ct.e**2 * n_p / (ct.m_e * ct.epsilon_0))
    
    k = 1.0 / (2 * np.pi * ct.e * dr * dxi * s_d * n_p)
    q_normalized = q_inv / k

    w_salame = q_normalized / bunch.q_species 
    bunch.w = w_salame

    print([np.max(q_bunch), np.min(q_bunch)])

    # clear b_t_bunch like your code does
    #b_t_bunch[:] = 0.0
    #q_bunch[:] = 0
