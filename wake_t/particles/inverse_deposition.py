"""
Inverse (adjoint-style) of deposit_3d_distribution: gather/interpolate from a
2D (z, r) deposition_array back to particle weights w at particle locations.

Important:
- This is NOT a true mathematical inverse of deposition (deposit is many-to-one).
- This is the PIC-consistent "inverse": gather using the SAME shape functions
  (linear/cubic) and SAME indexing/guard-cell conventions.

The interface mirrors deposit_3d_distribution:
inverse_deposit_3d_distribution -> chooses linear/cubic.

It is used in SALAME algorithm in physics_model/plasma_wakefields/qs_rz_baxevenis_ion/wakefield.py
"""

import math
import numpy as np

from wake_t.utilities.numba import njit_serial, prange
from wake_t.particles.deposition import deposit_3d_distribution

from scipy.sparse.linalg import LinearOperator, lsqr
from numpy.linalg import lstsq
from scipy.optimize import nnls
from scipy.optimize import lsq_linear


@njit_serial()
def inverse_deposit_3d_distribution(
    z,
    x,
    y,
    z_min,
    r_min,
    nz,
    nr,
    dz,
    dr,
    deposition_array,
    p_shape="cubic",
    use_ruyten=False,
    r_min_deposit=0.0,
):
    """
    Gather/interpolate from deposition_array onto particle positions.

    Parameters
    ----------
    z, x, y : arrays
        Particle coordinates.
    z_min, r_min, nz, nr, dz, dr : grid definition (same meaning as deposit_*).
    deposition_array : 2D array
        Size (nr+4, nz+4) in your docstring, but note your code indexes as
        deposition_array[iz, ir]. Keep the same layout you already use.
    p_shape : 'linear' or 'cubic'
    use_ruyten : bool
        Apply same Ruyten correction to the *radial* shape factors as in deposit.
        (This is the consistent adjoint counterpart.)
    r_min_deposit : float
        Minimum radius required to gather.

    Returns
    -------
    w : 1D array
        Gathered quantity at particle positions.
    bool
        Whether all particles were within bounds and successfully gathered.
    """
    if p_shape == "linear":
        return inverse_deposit_3d_distribution_linear(
            z, x, y,
            z_min, r_min, nz, nr, dz, dr,
            deposition_array,
            use_ruyten, r_min_deposit
        )
    elif p_shape == "cubic":
        return inverse_deposit_3d_distribution_cubic(
            z, x, y,
            z_min, r_min, nz, nr, dz, dr,
            deposition_array,
            use_ruyten, r_min_deposit
        )
    else:
        raise ValueError(
            "Particle shape not recognized. Possible values are 'linear' or 'cubic'."
        )


@njit_serial
def inverse_deposit_3d_distribution_linear(
    z,
    x,
    y,
    z_min,
    r_min,
    nz,
    nr,
    dz,
    dr,
    deposition_array,
    use_ruyten=False,
    r_min_deposit=0.0,
):
    """Gather (CIC/linear) from grid to particles. Returns (w, all_gathered)."""

    # Optional Ruyten coefficients (same as deposit_3d_distribution_linear)
    if use_ruyten:
        ruyten_coef = np.zeros(nr + 1)
        r_grid = (np.arange(nr) + 0.5) * dr  # cell-centered in r
        cell_volume = np.pi * dz * ((r_grid + 0.5 * dr) ** 2 - (r_grid - 0.5 * dr) ** 2)
        cell_volume_norm = cell_volume / (2 * np.pi * dr**2 * dz)
        cell_number = np.arange(nr) + 1
        ruyten_coef[1:] = (
            6.0
            / cell_number
            * (np.cumsum(cell_volume_norm) - 0.5 * cell_number**2 - 1.0 / 24.0)
        )

    z_max = z_min + (nz - 1) * dz
    r_max = nr * dr

    w_out = np.zeros(z.shape[0])
    all_gathered = True

    for i in prange(z.shape[0]):
        x_i = x[i]
        y_i = y[i]
        z_i = z[i]
        r_i = math.sqrt(x_i * x_i + y_i * y_i)

        if (z_i >= z_min) and (z_i <= z_max) and (r_i >= r_min_deposit) and (r_i <= r_max):
            r_cell = (r_i - r_min) / dr
            z_cell = (z_i - z_min) / dz

            # Same indexing convention as deposit_3d_distribution_linear
            ir_cell = min(int(math.ceil(r_cell)) + 1, nr + 2)
            iz_cell = int(math.ceil(z_cell)) + 1

            # u_r relative to left neighbor gridpoint in r
            if r_cell < 0.0:
                u_r = 1.0
            else:
                u_r = r_cell - int(math.ceil(r_cell)) + 1.0

            # u_z relative to left neighbor gridpoint in z
            if z_cell < 0.0:
                u_z = 1.0
            elif r_cell > nz - 1:  # NOTE: matches your deposit code (even though it looks odd)
                u_z = 0.0
            else:
                u_z = z_cell - int(math.ceil(z_cell)) + 1.0

            zsl_0 = 1.0 - u_z
            zsl_1 = u_z
            rsl_0 = 1.0 - u_r
            rsl_1 = u_r

            if use_ruyten:
                ir0 = min(int(math.ceil(r_cell)), nr)
                rc = ruyten_coef[ir0]
                corr = rc * (1.0 - u_r) * u_r
                rsl_0 += corr
                rsl_1 -= corr

            # Gather (adjoint of deposit): weighted sum of grid values
            w_i = 0.0
            w_i += zsl_0 * rsl_0 * deposition_array[iz_cell + 0, ir_cell + 0]
            w_i += zsl_0 * rsl_1 * deposition_array[iz_cell + 0, ir_cell + 1]
            w_i += zsl_1 * rsl_0 * deposition_array[iz_cell + 1, ir_cell + 0]
            w_i += zsl_1 * rsl_1 * deposition_array[iz_cell + 1, ir_cell + 1]
            w_out[i] = w_i
        else:
            all_gathered = False
            w_out[i] = 0.0

    return w_out, all_gathered


@njit_serial
def inverse_deposit_3d_distribution_cubic(
    z,
    x,
    y,
    z_min,
    r_min,
    nz,
    nr,
    dz,
    dr,
    deposition_array,
    use_ruyten=False,
    r_min_deposit=0.0,
):
    """Gather (3rd-order/cubic B-spline) from grid to particles. Returns (w, all_gathered)."""

    # Optional Ruyten coefficients (same as deposit_3d_distribution_cubic)
    if use_ruyten:
        ruyten_coef = np.zeros(nr + 1)
        r_grid = (np.arange(nr) + 0.5) * dr  # cell-centered in r
        cell_volume = np.pi * dz * ((r_grid + 0.5 * dr) ** 2 - (r_grid - 0.5 * dr) ** 2)
        cell_volume_norm = cell_volume / (2 * np.pi * dr**2 * dz)
        cell_number = np.arange(nr) + 1
        ruyten_coef[1:] = (
            6.0
            / cell_number
            * (np.cumsum(cell_volume_norm) - 0.5 * cell_number**2 - 0.125)
        )
        ruyten_coef[1] = 6.0 * (cell_volume_norm[0] - 0.5 - 239.0 / (15.0 * 2.0**7))

    z_max = z_min + (nz - 1) * dz
    r_max = nr * dr

    w_out = np.zeros(z.shape[0])
    all_gathered = True

    inv_6 = 1.0 / 6.0

    for i in prange(z.shape[0]):
        x_i = x[i]
        y_i = y[i]
        z_i = z[i]
        r_i = math.sqrt(x_i * x_i + y_i * y_i)

        if (z_i >= z_min) and (z_i <= z_max) and (r_i >= r_min_deposit) and (r_i <= r_max):
            r_cell = (r_i - r_min) / dr
            z_cell = (z_i - z_min) / dz

            # Same base indices as deposit_3d_distribution_cubic
            ir_cell = min(int(math.ceil(r_cell)), nr + 2)
            iz_cell = int(math.ceil(z_cell))

            u_z = z_cell - int(math.ceil(z_cell)) + 1.0
            u_r = r_cell - int(math.ceil(r_cell)) + 1.0

            v_z = 1.0 - u_z
            v_r = 1.0 - u_r

            # Cubic B-spline coefficients
            zsc_0 = inv_6 * v_z**3
            zsc_1 = inv_6 * (3.0 * u_z**3 - 6.0 * u_z**2 + 4.0)
            zsc_2 = inv_6 * (3.0 * v_z**3 - 6.0 * v_z**2 + 4.0)
            zsc_3 = inv_6 * u_z**3

            rsc_0 = inv_6 * v_r**3
            rsc_1 = inv_6 * (3.0 * u_r**3 - 6.0 * u_r**2 + 4.0)
            rsc_2 = inv_6 * (3.0 * v_r**3 - 6.0 * v_r**2 + 4.0)
            rsc_3 = inv_6 * u_r**3

            if use_ruyten:
                ir0 = min(int(math.ceil(r_cell)), nr)
                rc = ruyten_coef[ir0]
                corr = rc * v_r * u_r
                rsc_1 += corr
                rsc_2 -= corr

            # Apply the SAME boundary-folding logic as deposit (to stay consistent)
            if r_cell <= 0.0:
                rsc_3 += rsc_0
                rsc_2 += rsc_1
                rsc_0 = 0.0
                rsc_1 = 0.0
            elif r_cell <= 1.0:
                rsc_1 += rsc_0
                rsc_0 = 0.0

            if z_cell <= 0.0:
                zsc_3 += zsc_0
                zsc_2 += zsc_1
                zsc_0 = 0.0
                zsc_1 = 0.0
            elif z_cell <= 1.0:
                zsc_1 += zsc_0
                zsc_0 = 0.0
            elif z_cell > nz - 1:
                zsc_0 += zsc_3
                zsc_1 += zsc_2
                zsc_2 = 0.0
                zsc_3 = 0.0
            elif z_cell > nz - 2:
                zsc_2 += zsc_3
                zsc_3 = 0.0

            # Gather: 4x4 stencil
            w_i = 0.0

            # Unroll for speed & numba-friendliness
            # iz_cell + {0,1,2,3} and ir_cell + {0,1,2,3}
            w_i += zsc_0 * (rsc_0 * deposition_array[iz_cell + 0, ir_cell + 0] +
                            rsc_1 * deposition_array[iz_cell + 0, ir_cell + 1] +
                            rsc_2 * deposition_array[iz_cell + 0, ir_cell + 2] +
                            rsc_3 * deposition_array[iz_cell + 0, ir_cell + 3])
            w_i += zsc_1 * (rsc_0 * deposition_array[iz_cell + 1, ir_cell + 0] +
                            rsc_1 * deposition_array[iz_cell + 1, ir_cell + 1] +
                            rsc_2 * deposition_array[iz_cell + 1, ir_cell + 2] +
                            rsc_3 * deposition_array[iz_cell + 1, ir_cell + 3])
            w_i += zsc_2 * (rsc_0 * deposition_array[iz_cell + 2, ir_cell + 0] +
                            rsc_1 * deposition_array[iz_cell + 2, ir_cell + 1] +
                            rsc_2 * deposition_array[iz_cell + 2, ir_cell + 2] +
                            rsc_3 * deposition_array[iz_cell + 2, ir_cell + 3])
            w_i += zsc_3 * (rsc_0 * deposition_array[iz_cell + 3, ir_cell + 0] +
                            rsc_1 * deposition_array[iz_cell + 3, ir_cell + 1] +
                            rsc_2 * deposition_array[iz_cell + 3, ir_cell + 2] +
                            rsc_3 * deposition_array[iz_cell + 3, ir_cell + 3])

            w_out[i] = w_i
        else:
            all_gathered = False
            w_out[i] = 0.0

    return w_out, all_gathered









@njit_serial
def _build_ruyten_coef_cubic(nr, dr, dz):
    ruyten_coef = np.zeros(nr + 1)
    r_grid = (np.arange(nr) + 0.5) * dr  # cell-centered
    cell_volume = np.pi * dz * ((r_grid + 0.5 * dr) ** 2 - (r_grid - 0.5 * dr) ** 2)
    cell_volume_norm = cell_volume / (2 * np.pi * dr**2 * dz)
    cell_number = np.arange(nr) + 1
    ruyten_coef[1:] = (
        6.0
        / cell_number
        * (np.cumsum(cell_volume_norm) - 0.5 * cell_number**2 - 0.125)
    )
    ruyten_coef[1] = 6.0 * (cell_volume_norm[0] - 0.5 - 239.0 / (15 * 2**7))
    return ruyten_coef


@njit_serial
def _cubic_coeffs(u):
    inv_6 = 1.0 / 6.0
    v = 1.0 - u
    sc0 = inv_6 * v**3
    sc1 = inv_6 * (3.0 * u**3 - 6.0 * u**2 + 4.0)
    sc2 = inv_6 * (3.0 * v**3 - 6.0 * v**2 + 4.0)
    sc3 = inv_6 * u**3
    return sc0, sc1, sc2, sc3


@njit_serial(parallel=True)
def precompute_stencil_cubic_ruyten(
    z, x, y,
    z_min, r_min, nz, nr, dz, dr,
    use_ruyten, r_min_deposit,
    nz_tot, nr_tot,
):
    """
    Precompute for each particle:
      - 16 flattened grid indices (row-major on [iz, ir])
      - 16 corresponding shape coefficients

    Indices refer to the full deposition array (including guard cells).
    """
    Np = z.shape[0]
    idx = -np.ones((Np, 16), dtype=np.int64)
    coef = np.zeros((Np, 16), dtype=np.float64)
    active = np.zeros(Np, dtype=np.bool_)

    z_max = z_min + (nz - 1) * dz
    r_max = nr * dr

    ruyten_coef = np.zeros(1)
    if use_ruyten:
        ruyten_coef = _build_ruyten_coef_cubic(nr, dr, dz)

    for i in prange(Np):
        zi = z[i]
        ri = math.sqrt(x[i]*x[i] + y[i]*y[i])

        if not (zi >= z_min and zi <= z_max and ri >= r_min_deposit and ri <= r_max):
            continue

        z_cell = (zi - z_min) / dz
        r_cell = (ri - r_min) / dr

        # --- Wake-T cubic indexing convention ---
        iz0 = int(math.ceil(z_cell))
        ir0 = min(int(math.ceil(r_cell)), nr + 2)

        uz = z_cell - math.ceil(z_cell) + 1.0
        ur = r_cell - math.ceil(r_cell) + 1.0

        zsc0, zsc1, zsc2, zsc3 = _cubic_coeffs(uz)
        rsc0, rsc1, rsc2, rsc3 = _cubic_coeffs(ur)

        # Ruyten correction (exactly as deposit)
        if use_ruyten:
            ir = min(int(math.ceil(r_cell)), nr)
            rc = ruyten_coef[ir]
            vr = 1.0 - ur
            rsc1 += rc * vr * ur
            rsc2 -= rc * vr * ur

        # Axis folding (exactly as deposit)
        if r_cell <= 0.0:
            rsc3 += rsc0
            rsc2 += rsc1
            rsc0 = 0.0
            rsc1 = 0.0
        elif r_cell <= 1.0:
            rsc1 += rsc0
            rsc0 = 0.0

        # z-boundary folding (exactly as deposit)
        if z_cell <= 0.0:
            zsc3 += zsc0
            zsc2 += zsc1
            zsc0 = 0.0
            zsc1 = 0.0
        elif z_cell <= 1.0:
            zsc1 += zsc0
            zsc0 = 0.0
        elif z_cell > nz - 1:
            zsc0 += zsc3
            zsc1 += zsc2
            zsc2 = 0.0
            zsc3 = 0.0
        elif z_cell > nz - 2:
            zsc2 += zsc3
            zsc3 = 0.0

        zc = (zsc0, zsc1, zsc2, zsc3)
        rc = (rsc0, rsc1, rsc2, rsc3)

        # Store 16 entries (flattened index = iz*nr_tot + ir)
        k = 0
        ok = True
        for a in range(4):
            iz = iz0 + a
            if iz < 0 or iz >= nz_tot:
                ok = False
                break
            for b in range(4):
                ir = ir0 + b
                if ir < 0 or ir >= nr_tot:
                    ok = False
                    break
                idx[i, k] = iz * nr_tot + ir
                coef[i, k] = zc[a] * rc[b]
                k += 1
            if not ok:
                break

        if ok:
            active[i] = True

    return idx, coef, active


@njit_serial(parallel=True)
def gather_from_grid(idx, coef, active, grid_flat):
    """Compute (A^T grid_flat)_i using the precomputed stencil."""
    Np = idx.shape[0]
    out = np.zeros(Np, dtype=np.float64)
    for i in prange(Np):
        if not active[i]:
            continue
        s = 0.0
        for k in range(16):
            s += coef[i, k] * grid_flat[idx[i, k]]
        out[i] = s
    return out


def inverse_deposit_fast(
    z, x, y,
    z_min, r_min, nz, nr, dz, dr,
    rho0,
    *,
    use_ruyten=True,
    r_min_deposit=0.0,
    sign_mode="preserve_sign_of_init",   # "nonnegative" or "preserve_sign_of_init" or "none"
    n_iter=20,
    alpha=1.0,
    eps=1e-30,
):
    """
    Fast approximate inverse for cubic+Ruyten using:
      - forward: deposit_3d_distribution (Wake-T)
      - adjoint: gather using precomputed cubic+Ruyten stencils
      - normalization: count_p trick

    Returns w_inv, info
    """
    z = np.asarray(z, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rho0 = np.asarray(rho0, dtype=np.float64)

    nz_tot, nr_tot = rho0.shape
    assert nz_tot == nz + 4 and nr_tot == nr + 4, "rho0 must be (nz+4, nr+4)"

    # Precompute stencils once
    idx, coef, active = precompute_stencil_cubic_ruyten(
        z, x, y, z_min, r_min, nz, nr, dz, dr,
        use_ruyten, r_min_deposit, nz_tot, nr_tot
    )

    # count_p = A * 1
    count_p = np.zeros_like(rho0)
    ones = np.ones_like(z)
    deposit_3d_distribution(
        z, x, y, ones,
        z_min, r_min, nz, nr, dz, dr,
        count_p,
        p_shape="cubic",
        use_ruyten=use_ruyten,
        r_min_deposit=r_min_deposit,
    )

    # Initial guess: backprojection of normalized rho0
    rho0_flat = rho0.reshape(-1)
    count_flat = count_p.reshape(-1)
    norm_grid = rho0_flat / (count_flat + eps)
    w = gather_from_grid(idx, coef, active, norm_grid)

    # Set sign constraint baseline if needed
    #if sign_mode == "nonnegative":
    #    w = np.maximum(w, 0.0)
    #elif sign_mode == "preserve_sign_of_init":
    #    sgn0 = np.sign(w)
    #    sgn0[sgn0 == 0] = 1.0
    #else:
    #    sgn0 = None

    grid_sign = np.sign(np.sum(rho0))
    
    # after each update step:
    if sign_mode == "fixed_sign":
        w = grid_sign * np.abs(w)
    elif sign_mode == "nonnegative":
        w = np.maximum(w, 0.0)
    elif sign_mode == "preserve_sign_of_init":
        sgn0 = np.sign(w)
        sgn0[sgn0 == 0] = 1.0
    else:
        sgn0 = None


    # Work buffers
    rho = np.zeros_like(rho0)
    res = np.zeros_like(rho0)

    for it in range(n_iter):
        rho.fill(0.0)
        deposit_3d_distribution(
            z, x, y, w,
            z_min, r_min, nz, nr, dz, dr,
            rho,
            p_shape="cubic",
            use_ruyten=use_ruyten,
            r_min_deposit=r_min_deposit,
        )

        # residual on grid
        res[:] = rho0 - rho
        res_flat = res.reshape(-1)

        # normalized residual (count_p trick)
        grad_grid = res_flat / (count_flat + eps)

        # approximate gradient step using adjoint gather
        g = gather_from_grid(idx, coef, active, grad_grid)
        w = w + alpha * g

        # project back to sign constraint
        if sign_mode == "nonnegative":
            w = np.maximum(w, 0.0)
        elif sign_mode == "preserve_sign_of_init":
            w = np.abs(w) * sgn0

    info = {
        "n_iter": int(n_iter),
        "alpha": float(alpha),
        "use_ruyten": bool(use_ruyten),
        "r_min_deposit": float(r_min_deposit),
        "active_frac": float(np.mean(active)),
    }
    return w, info






def inverse_deposit_lsqr(
    z, x, y,
    z_min, r_min, nz, nr, dz, dr,
    rho0,
    *,
    use_ruyten=True,
    r_min_deposit=0.0,
    mask=None,                 # if None, uses count_grid>0
    damp=0.0,                  # Tikhonov damping for stability
    atol=1e-12,
    btol=1e-12,
    iter_lim=200,
    enforce_total=True,
    fixed_sign=None,           # None, or -1/+1
):
    """
    Solve min ||A w - rho0||_2 using LSQR without building A.

    A w computed by deposit_3d_distribution.
    A^T u computed by gather_from_grid (using precomputed stencils).

    Optionally restrict to rows where count_grid>0 (mask) for better conditioning.
    """

    z = np.asarray(z, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rho0 = np.asarray(rho0, dtype=np.float64)

    nz_tot, nr_tot = rho0.shape
    assert nz_tot == nz + 4 and nr_tot == nr + 4, "rho0 must be (nz+4, nr+4)"

    # Precompute particle stencils once (cubic+Ruyten+folding consistent with deposit)
    idx, coef, active = precompute_stencil_cubic_ruyten(
        z, x, y, z_min, r_min, nz, nr, dz, dr,
        use_ruyten, r_min_deposit, nz_tot, nr_tot
    )

    # Build count_grid once (for mask and diagnostics)
    count_grid = np.zeros_like(rho0)
    ones = np.ones_like(z)
    deposit_3d_distribution(
        z, x, y, ones,
        z_min, r_min, nz, nr, dz, dr,
        count_grid,
        p_shape="cubic",
        use_ruyten=use_ruyten,
        r_min_deposit=r_min_deposit,
    )

    if mask is None:
        mask = count_grid > 0

    # Compress b to masked rows
    b_full = rho0.reshape(-1)
    mask_flat = mask.reshape(-1)
    b = b_full[mask_flat]
    m = b.size
    n = z.size

    # Work buffers to avoid reallocations
    rho_tmp = np.zeros_like(rho0)
    u_full = np.zeros_like(b_full)  # full-grid vector for rmatvec scatter

    def matvec(w):
        # y = A w, but return only masked entries
        rho_tmp.fill(0.0)
        deposit_3d_distribution(
            z, x, y, w,
            z_min, r_min, nz, nr, dz, dr,
            rho_tmp,
            p_shape="cubic",
            use_ruyten=use_ruyten,
            r_min_deposit=r_min_deposit,
        )
        y_full = rho_tmp.reshape(-1)
        return y_full[mask_flat]

    def rmatvec(u):
        # x = A^T u, where u is masked-grid vector
        u_full.fill(0.0)
        u_full[mask_flat] = u
        return gather_from_grid(idx, coef, active, u_full)

    Aop = LinearOperator(shape=(m, n), matvec=matvec, rmatvec=rmatvec, dtype=np.float64)

    sol = lsqr(Aop, b, damp=damp, atol=atol, btol=btol, iter_lim=iter_lim)
    w = sol[0]

    # Optional fixed-sign projection (do AFTER LSQR; projecting during solve hurts residual)
    if fixed_sign in (-1, 1):
        w = fixed_sign * np.abs(w)

    # Enforce total charge exactly (recommended in your SALAME workflow)
    if enforce_total:
        target_sum = float(np.sum(rho0))  # guards are zero; OK
        inv_sum = float(np.sum(w))
        if inv_sum != 0.0:
            w *= (target_sum / inv_sum)

    info = {
        "istop": int(sol[1]),
        "itn": int(sol[2]),
        "r1norm": float(sol[3]),
        "r2norm": float(sol[4]),
        "anorm": float(sol[5]),
        "acond": float(sol[6]),
        "arnorm": float(sol[7]),
        "xnorm": float(sol[8]),
        "active_particle_frac": float(np.mean(active)),
        "active_grid_frac": float(np.mean(mask)),
        "mask_rows": int(m),
        "n_part": int(n),
    }
    return w, info, count_grid, mask





#def inverse_line_deposition_exact(
#    xi, x, y,
#    xi_min, r_min,
#    n_xi, n_r, dxi, dr,
#    lambda_target,
#    *,
#    p_shape="cubic",
#    use_ruyten=True,
#    r_min_deposit=0.0,
#):
#    """
#    Invert only the line charge lambda(xi), not the full 2D grid.
#    """
#
#    xi = np.asarray(xi, dtype=np.float64)
#    x = np.asarray(x, dtype=np.float64)
#    y = np.asarray(y, dtype=np.float64)
#    lambda_target = np.asarray(lambda_target, dtype=np.float64)
#
#    Np = xi.size
#    A = np.zeros((n_xi, Np), dtype=np.float64)
#
#    tmp = np.zeros((n_xi + 4, n_r + 4), dtype=np.float64)
#    one = np.ones(1, dtype=np.float64)
#
#    for i in range(Np):
#        tmp.fill(0.0)
#        deposit_3d_distribution(
#            xi[i:i+1], x[i:i+1], y[i:i+1],
#            one,
#            xi_min, r_min,
#            n_xi, n_r,
#            dxi, dr,
#            tmp,
#            p_shape=p_shape,
#            use_ruyten=use_ruyten,
#            r_min_deposit=r_min_deposit,
#        )
#        # sum over r interior only
#        A[:, i] = np.sum(tmp[2:-2, 2:-2], axis=1)
#
#    q_inv, *_ = lstsq(A, lambda_target, rcond=None)
#
#    # enforce exact total
#    s_tgt = np.sum(lambda_target)
#    s_inv = np.sum(q_inv)
#    if s_inv != 0.0:
#        q_inv *= s_tgt / s_inv
#
#    return q_inv






def inverse_line_deposition_exact(
    xi, x, y,
    xi_min, r_min,
    n_xi, n_r, dxi, dr,
    lambda_target,
    *,
    p_shape="cubic",
    use_ruyten=True,
    r_min_deposit=0.0,
):
    """
    Invert only the line charge lambda(xi), not the full 2D grid,
    with the constraint q_inv <= 0 for all particles.
    """

    xi = np.asarray(xi, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lambda_target = np.asarray(lambda_target, dtype=np.float64)

    Np = xi.size
    A = np.zeros((n_xi, Np), dtype=np.float64)

    tmp = np.zeros((n_xi + 4, n_r + 4), dtype=np.float64)
    one = np.ones(1, dtype=np.float64)

    # Build 1D line-deposition matrix
    for i in range(Np):
        tmp.fill(0.0)
        deposit_3d_distribution(
            xi[i:i+1], x[i:i+1], y[i:i+1],
            one,
            xi_min, r_min,
            n_xi, n_r,
            dxi, dr,
            tmp,
            p_shape=p_shape,
            use_ruyten=use_ruyten,
            r_min_deposit=r_min_deposit,
        )
        A[:, i] = np.sum(tmp[2:-2, 2:-2], axis=1)

    # Solve A u ≈ -lambda_target with u >= 0
    u, rnorm = nnls(A, -lambda_target)

    # Convert back to q_inv <= 0
    q_inv = -u

    #res = lsq_linear(A, lambda_target, bounds=(-np.inf, 0.0), lsmr_tol='auto')
    #q_inv = res.x
    
    # Enforce exact total charge while keeping q_inv <= 0
    s_tgt = np.sum(lambda_target)
    s_inv = np.sum(q_inv)
    if s_inv != 0.0:
        scale = s_tgt / s_inv
        if scale > 0.0:
            q_inv *= scale

    return q_inv






import math
import numpy as np
from scipy.sparse import coo_matrix
from scipy.optimize import lsq_linear


def _ruyten_linear(nr, dr, dz):
    ruyten_coef = np.zeros(nr + 1)
    r_grid = (np.arange(nr) + 0.5) * dr
    cell_volume = np.pi * dz * ((r_grid + 0.5 * dr) ** 2 - (r_grid - 0.5 * dr) ** 2)
    cell_volume_norm = cell_volume / (2 * np.pi * dr**2 * dz)
    cell_number = np.arange(nr) + 1
    ruyten_coef[1:] = (
        6.0 / cell_number
        * (np.cumsum(cell_volume_norm) - 0.5 * cell_number**2 - 1.0 / 24.0)
    )
    return ruyten_coef


def _ruyten_cubic(nr, dr, dz):
    ruyten_coef = np.zeros(nr + 1)
    r_grid = (np.arange(nr) + 0.5) * dr
    cell_volume = np.pi * dz * ((r_grid + 0.5 * dr) ** 2 - (r_grid - 0.5 * dr) ** 2)
    cell_volume_norm = cell_volume / (2 * np.pi * dr**2 * dz)
    cell_number = np.arange(nr) + 1
    ruyten_coef[1:] = (
        6.0 / cell_number
        * (np.cumsum(cell_volume_norm) - 0.5 * cell_number**2 - 0.125)
    )
    ruyten_coef[1] = 6.0 * (cell_volume_norm[0] - 0.5 - 239.0 / (15 * 2**7))
    return ruyten_coef


def _z_stencil_linear(z_cell, nz):
    cz = int(math.ceil(z_cell))
    iz0 = cz + 1  # full-array base index
    if z_cell < 0:
        u = 1.0
    elif z_cell > nz - 1:
        u = 0.0
    else:
        u = z_cell - cz + 1.0
    c0 = 1.0 - u
    c1 = u
    return [iz0 + 0, iz0 + 1], [c0, c1]


def _r_stencil_linear(r_cell, nr, use_ruyten, ruyten_coef):
    cr = int(math.ceil(r_cell))
    ir0 = min(cr + 1, nr + 2)  # full-array base index
    if r_cell < 0:
        u = 1.0
    else:
        u = r_cell - cr + 1.0

    c0 = 1.0 - u
    c1 = u

    if use_ruyten:
        ir = min(int(math.ceil(r_cell)), nr)
        rc = ruyten_coef[ir]
        corr = (1.0 - u) * u
        c0 += rc * corr
        c1 -= rc * corr

    return [ir0 + 0, ir0 + 1], [c0, c1]


def _z_stencil_cubic(z_cell, nz):
    cz = int(math.ceil(z_cell))
    iz0 = cz  # full-array base index
    u = z_cell - cz + 1.0
    v = 1.0 - u
    inv_6 = 1.0 / 6.0

    c0 = inv_6 * v**3
    c1 = inv_6 * (3.0 * u**3 - 6.0 * u**2 + 4.0)
    c2 = inv_6 * (3.0 * v**3 - 6.0 * v**2 + 4.0)
    c3 = inv_6 * u**3

    if z_cell <= 0.0:
        c3 += c0
        c2 += c1
        c0 = 0.0
        c1 = 0.0
    elif z_cell <= 1.0:
        c1 += c0
        c0 = 0.0
    elif z_cell > nz - 1:
        c0 += c3
        c1 += c2
        c2 = 0.0
        c3 = 0.0
    elif z_cell > nz - 2:
        c2 += c3
        c3 = 0.0

    return [iz0 + 0, iz0 + 1, iz0 + 2, iz0 + 3], [c0, c1, c2, c3]


def _r_stencil_cubic(r_cell, nr, use_ruyten, ruyten_coef):
    cr = int(math.ceil(r_cell))
    ir0 = min(cr, nr + 2)  # full-array base index
    u = r_cell - cr + 1.0
    v = 1.0 - u
    inv_6 = 1.0 / 6.0

    c0 = inv_6 * v**3
    c1 = inv_6 * (3.0 * u**3 - 6.0 * u**2 + 4.0)
    c2 = inv_6 * (3.0 * v**3 - 6.0 * v**2 + 4.0)
    c3 = inv_6 * u**3

    if use_ruyten:
        ir = min(int(math.ceil(r_cell)), nr)
        rc = ruyten_coef[ir]
        corr = v * u
        c1 += rc * corr
        c2 -= rc * corr

    if r_cell <= 0.0:
        c3 += c0
        c2 += c1
        c0 = 0.0
        c1 = 0.0
    elif r_cell <= 1.0:
        c1 += c0
        c0 = 0.0

    return [ir0 + 0, ir0 + 1, ir0 + 2, ir0 + 3], [c0, c1, c2, c3]


def build_line_deposition_matrix_sparse(
    xi, x, y,
    xi_min, r_min,
    n_xi, n_r, dxi, dr,
    *,
    p_shape="cubic",
    use_ruyten=True,
    r_min_deposit=0.0,
):
    """
    Build sparse matrix A such that:
        lambda = A @ q_part
    where lambda is the interior line charge (size n_xi).
    """
    xi = np.asarray(xi, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    z_max = xi_min + (n_xi - 1) * dxi
    r_max = n_r * dr

    if p_shape == "linear":
        ruyten_coef = _ruyten_linear(n_r, dr, dxi) if use_ruyten else None
    elif p_shape == "cubic":
        ruyten_coef = _ruyten_cubic(n_r, dr, dxi) if use_ruyten else None
    else:
        raise ValueError("p_shape must be 'linear' or 'cubic'")

    rows = []
    cols = []
    data = []

    for i in range(xi.size):
        zi = xi[i]
        ri = math.sqrt(x[i] ** 2 + y[i] ** 2)

        if not (zi >= xi_min and zi <= z_max and ri >= r_min_deposit and ri <= r_max):
            continue

        z_cell = (zi - xi_min) / dxi
        r_cell = (ri - r_min) / dr

        if p_shape == "linear":
            z_idx_full, z_coef = _z_stencil_linear(z_cell, n_xi)
            r_idx_full, r_coef = _r_stencil_linear(r_cell, n_r, use_ruyten, ruyten_coef)
        else:
            z_idx_full, z_coef = _z_stencil_cubic(z_cell, n_xi)
            r_idx_full, r_coef = _r_stencil_cubic(r_cell, n_r, use_ruyten, ruyten_coef)

        # Sum only interior radial contributions (2:-2)
        s_r = 0.0
        for ir_full, cr in zip(r_idx_full, r_coef):
            if 2 <= ir_full < n_r + 2:
                s_r += cr

        if s_r == 0.0:
            continue

        for iz_full, cz in zip(z_idx_full, z_coef):
            if cz == 0.0:
                continue
            if 2 <= iz_full < n_xi + 2:
                rows.append(iz_full - 2)   # interior xi row
                cols.append(i)
                data.append(cz * s_r)

    A = coo_matrix((data, (rows, cols)), shape=(n_xi, xi.size)).tocsr()
    return A


def inverse_line_deposition_negative_sparse(
    xi, x, y,
    xi_min, r_min,
    n_xi, n_r, dxi, dr,
    lambda_target,
    *,
    p_shape="cubic",
    use_ruyten=True,
    r_min_deposit=0.0,
    tol=1e-12,
    max_iter=200,
    clip_target_positive=False,
):
    """
    Fast sparse bounded solve for q_inv <= 0.

    Solves:
        minimize ||A q - lambda_target||_2
        subject to q <= 0
    """
    lambda_target = np.asarray(lambda_target, dtype=np.float64).copy()

    if clip_target_positive:
        lambda_target = np.minimum(lambda_target, 0.0)

    # Feasibility check
    lam_max = float(np.max(lambda_target))
    if lam_max > 1e-14:
        print("Warning: lambda_target has positive entries; exact q_inv<=0 fit is impossible.")
        print("lambda_target max =", lam_max)

    A = build_line_deposition_matrix_sparse(
        xi, x, y,
        xi_min, r_min,
        n_xi, n_r, dxi, dr,
        p_shape=p_shape,
        use_ruyten=use_ruyten,
        r_min_deposit=r_min_deposit,
    )

    res = lsq_linear(
        A,
        lambda_target,
        bounds=(-np.inf, 0.0),
        method="trf",
        lsmr_tol=tol,
        tol=tol,
        max_iter=max_iter,
    )

    q_inv = res.x

    # Enforce exact total while preserving negativity
    s_tgt = float(np.sum(lambda_target))
    s_inv = float(np.sum(q_inv))
    if s_inv != 0.0:
        scale = s_tgt / s_inv
        if scale > 0.0:
            q_inv *= scale

    # Safety clamp for roundoff
    q_inv = np.minimum(q_inv, 0.0)

    info = {
        "success": res.success,
        "status": int(res.status),
        "message": res.message,
        "cost": float(res.cost),
        "optimality": float(res.optimality),
        "lambda_min": float(np.min(lambda_target)),
        "lambda_max": float(np.max(lambda_target)),
        "nnz": int(A.nnz),
        "shape": A.shape,
    }
    return q_inv, info
