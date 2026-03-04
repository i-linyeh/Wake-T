import math
import numpy as np
import scipy.constants as ct

from wake_t.utilities.numba import njit_serial, prange


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
    This is the PIC-consistent adjoint of Wake-T deposit_3d_distribution.

    IMPORTANT:
    - deposition_array is assumed to have shape (nz+4, nr+4) and be indexed [iz, ir],
      consistent with the deposit_3d_distribution code you pasted.
    - No cell-volume normalization is applied here. This gathers in the same "units"
      as deposition_array stores.
    """
    if p_shape == "linear":
        return inverse_deposit_3d_distribution_linear(
            z, x, y, z_min, r_min, nz, nr, dz, dr,
            deposition_array, use_ruyten, r_min_deposit
        )
    elif p_shape == "cubic":
        return inverse_deposit_3d_distribution_cubic(
            z, x, y, z_min, r_min, nz, nr, dz, dr,
            deposition_array, use_ruyten, r_min_deposit
        )
    else:
        raise ValueError("p_shape must be 'linear' or 'cubic'.")


@njit_serial
def inverse_deposit_3d_distribution_linear(
    z, x, y,
    z_min, r_min, nz, nr, dz, dr,
    deposition_array,
    use_ruyten=False,
    r_min_deposit=0.0,
):
    """Gather (linear/CIC) from (nz+4,nr+4) grid to particles."""

    # Ruyten coefficients (same expression as deposit_3d_distribution_linear)
    if use_ruyten:
        ruyten_coef = np.zeros(nr + 1)
        r_grid = (np.arange(nr) + 0.5) * dr
        cell_volume = np.pi * dz * ((r_grid + 0.5 * dr) ** 2 - (r_grid - 0.5 * dr) ** 2)
        cell_volume_norm = cell_volume / (2 * np.pi * dr**2 * dz)
        cell_number = np.arange(nr) + 1
        ruyten_coef[1:] = (
            6.0 / cell_number * (np.cumsum(cell_volume_norm) - 0.5 * cell_number**2 - 1.0 / 24.0)
        )

    z_max = z_min + (nz - 1) * dz
    r_max = nr * dr

    w_out = np.zeros(z.shape[0])
    all_gathered = True

    for i in prange(z.shape[0]):
        r_i = math.sqrt(x[i]*x[i] + y[i]*y[i])
        z_i = z[i]

        if (z_i >= z_min) and (z_i <= z_max) and (r_i >= r_min_deposit) and (r_i <= r_max):
            r_cell = (r_i - r_min) / dr
            z_cell = (z_i - z_min) / dz

            # same indexing as deposit linear
            ir_cell = min(int(math.ceil(r_cell)) + 1, nr + 2)
            iz_cell = int(math.ceil(z_cell)) + 1

            # same u_r as deposit linear
            if r_cell < 0.0:
                u_r = 1.0
            else:
                u_r = r_cell - int(math.ceil(r_cell)) + 1.0

            # IMPORTANT: fix the Wake-T typo here (deposit uses r_cell by mistake)
            if z_cell < 0.0:
                u_z = 1.0
            elif z_cell > nz - 1:
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
    z, x, y,
    z_min, r_min, nz, nr, dz, dr,
    deposition_array,
    use_ruyten=False,
    r_min_deposit=0.0,
):
    """Gather (cubic B-spline) from (nz+4,nr+4) grid to particles."""

    # Ruyten coefficients (same expression as deposit_3d_distribution_cubic)
    if use_ruyten:
        ruyten_coef = np.zeros(nr + 1)
        r_grid = (np.arange(nr) + 0.5) * dr
        cell_volume = np.pi * dz * ((r_grid + 0.5 * dr) ** 2 - (r_grid - 0.5 * dr) ** 2)
        cell_volume_norm = cell_volume / (2 * np.pi * dr**2 * dz)
        cell_number = np.arange(nr) + 1
        ruyten_coef[1:] = (
            6.0 / cell_number * (np.cumsum(cell_volume_norm) - 0.5 * cell_number**2 - 0.125)
        )
        ruyten_coef[1] = 6.0 * (cell_volume_norm[0] - 0.5 - 239.0 / (15.0 * 2.0**7))

    z_max = z_min + (nz - 1) * dz
    r_max = nr * dr

    w_out = np.zeros(z.shape[0])
    all_gathered = True

    inv_6 = 1.0 / 6.0

    for i in prange(z.shape[0]):
        r_i = math.sqrt(x[i]*x[i] + y[i]*y[i])
        z_i = z[i]

        if (z_i >= z_min) and (z_i <= z_max) and (r_i >= r_min_deposit) and (r_i <= r_max):
            r_cell = (r_i - r_min) / dr
            z_cell = (z_i - z_min) / dz

            # same base indices as deposit cubic
            ir_cell = min(int(math.ceil(r_cell)), nr + 2)
            iz_cell = int(math.ceil(z_cell))

            u_z = z_cell - int(math.ceil(z_cell)) + 1.0
            u_r = r_cell - int(math.ceil(r_cell)) + 1.0
            v_z = 1.0 - u_z
            v_r = 1.0 - u_r

            zsc_0 = inv_6 * v_z**3
            zsc_1 = inv_6 * (3.0*u_z**3 - 6.0*u_z**2 + 4.0)
            zsc_2 = inv_6 * (3.0*v_z**3 - 6.0*v_z**2 + 4.0)
            zsc_3 = inv_6 * u_z**3

            rsc_0 = inv_6 * v_r**3
            rsc_1 = inv_6 * (3.0*u_r**3 - 6.0*u_r**2 + 4.0)
            rsc_2 = inv_6 * (3.0*v_r**3 - 6.0*v_r**2 + 4.0)
            rsc_3 = inv_6 * u_r**3

            if use_ruyten:
                ir0 = min(int(math.ceil(r_cell)), nr)
                rc = ruyten_coef[ir0]
                corr = rc * v_r * u_r
                rsc_1 += corr
                rsc_2 -= corr

            # boundary folding (must match deposit cubic)
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

            w_i = 0.0
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


def inverse_deposit_bunch_charge(
    x, y, z,
    n_p,
    n_r, n_xi,
    r_grid, xi_grid,
    dr, dxi,
    p_shape,
    q_bunch,
    r_min_deposit=0.0,
):
    """
    Inverse of deposit_bunch_charge in the PIC-adjoint sense.

    Returns
    -------
    q_rec : ndarray
        Reconstructed particle charges [C] (same units as input q in deposit_bunch_charge).
    all_gathered : bool
    w_rec : ndarray
        The gathered normalized weights w = q*k (useful for debugging).
    k : float
        The normalization factor used in deposit_bunch_charge.
    """
    n_part = x.shape[0]
    s_d = ct.c / np.sqrt(ct.e**2 * n_p / (ct.m_e * ct.epsilon_0))
    k = 1.0 / (2 * np.pi * ct.e * dr * dxi * s_d * n_p)

    w_rec, all_gathered = inverse_deposit_3d_distribution(
        z=z, x=x, y=y,
        z_min=xi_grid[0],
        r_min=r_grid[0],
        nz=n_xi,
        nr=n_r,
        dz=dxi,
        dr=dr,
        deposition_array=q_bunch,
        p_shape=p_shape,
        use_ruyten=True,
        r_min_deposit=r_min_deposit,
    )

    # invert normalization: w = q*k  -> q = w/k
    q_rec = w_rec / k
    return q_rec, all_gathered, w_rec, k
