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
            use_ruyten,
            r_min_deposit,
        )
    elif p_shape == "cubic":
        return inverse_deposit_3d_distribution_cubic(
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
            use_ruyten,
            r_min_deposit,
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

        if (
            (z_i >= z_min)
            and (z_i <= z_max)
            and (r_i >= r_min_deposit)
            and (r_i <= r_max)
        ):
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
            elif (
                r_cell > nz - 1
            ):  # NOTE: matches your deposit code (even though it looks odd)
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

        if (
            (z_i >= z_min)
            and (z_i <= z_max)
            and (r_i >= r_min_deposit)
            and (r_i <= r_max)
        ):
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
            w_i += zsc_0 * (
                rsc_0 * deposition_array[iz_cell + 0, ir_cell + 0]
                + rsc_1 * deposition_array[iz_cell + 0, ir_cell + 1]
                + rsc_2 * deposition_array[iz_cell + 0, ir_cell + 2]
                + rsc_3 * deposition_array[iz_cell + 0, ir_cell + 3]
            )
            w_i += zsc_1 * (
                rsc_0 * deposition_array[iz_cell + 1, ir_cell + 0]
                + rsc_1 * deposition_array[iz_cell + 1, ir_cell + 1]
                + rsc_2 * deposition_array[iz_cell + 1, ir_cell + 2]
                + rsc_3 * deposition_array[iz_cell + 1, ir_cell + 3]
            )
            w_i += zsc_2 * (
                rsc_0 * deposition_array[iz_cell + 2, ir_cell + 0]
                + rsc_1 * deposition_array[iz_cell + 2, ir_cell + 1]
                + rsc_2 * deposition_array[iz_cell + 2, ir_cell + 2]
                + rsc_3 * deposition_array[iz_cell + 2, ir_cell + 3]
            )
            w_i += zsc_3 * (
                rsc_0 * deposition_array[iz_cell + 3, ir_cell + 0]
                + rsc_1 * deposition_array[iz_cell + 3, ir_cell + 1]
                + rsc_2 * deposition_array[iz_cell + 3, ir_cell + 2]
                + rsc_3 * deposition_array[iz_cell + 3, ir_cell + 3]
            )

            w_out[i] = w_i
        else:
            all_gathered = False
            w_out[i] = 0.0

    return w_out, all_gathered
