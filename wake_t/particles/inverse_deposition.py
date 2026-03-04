"""
PIC-consistent inverse (adjoint-style) of deposit_3d_distribution for a
cylindrical (z,r) grid.

Key point vs your current version:
- deposit_3d_distribution stores CELL-INTEGRATED quantity (e.g. charge per cell) in deposition_array.
- a PIC-consistent gather that returns a particle "weight" should sample the corresponding
  CELL-AVERAGED DENSITY, i.e. divide by the cylindrical cell volume V_r before applying
  the shape factors.

So we gather:
    w_i = sum_{stencil} S_z * S_r * (deposition_array / V_r)

This makes inverse+deposit behave sensibly and match analytic density profiles.

Array layout:
- deposition_array is indexed as deposition_array[iz, ir] and must have shape (nz+4, nr+4),
  i.e. SAME as Wake-T deposit_* uses in the code you pasted (docstring in Wake-T is wrong).
"""

import math
import numpy as np

from wake_t.utilities.numba import njit_serial, prange


@njit_serial()
def _build_cyl_cell_volumes(nr, dr, dz):
    """
    Cell volumes for annular cylindrical cells (cell-centered r grid).

    Cell j (0..nr-1) spans r in [j*dr, (j+1)*dr], so:
        V_j = pi * (( (j+1)dr )^2 - ( j dr )^2) * dz
            = pi * (2j+1) * dr^2 * dz

    Returns v_cell with length nr, corresponding to interior radial cells.
    """
    v = np.empty(nr)
    for j in range(nr):
        v[j] = math.pi * (2.0 * j + 1.0) * dr * dr * dz
    return v


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
    input_is_cell_integrated=True,
):
    """
    Gather/interpolate from deposition_array onto particle positions.

    Parameters
    ----------
    z, x, y : arrays
        Particle coordinates.
    z_min, r_min, nz, nr, dz, dr : grid definition (same meaning as deposit_*).
    deposition_array : 2D array
        Must be indexed as deposition_array[iz, ir] and have shape (nz+4, nr+4).
        If input_is_cell_integrated=True (default), values are cell-integrated
        (e.g. charge per cell). We convert to density by dividing by cell volume.
        If False, values are already density, and we do not divide by volume.
    p_shape : 'linear' or 'cubic'
    use_ruyten : bool
        Apply same Ruyten correction to radial shape factors as in deposit.
    r_min_deposit : float
        Minimum radius required to gather.
    input_is_cell_integrated : bool
        True if deposition_array stores integrated quantity per cell (Wake-T deposit does).
        False if deposition_array stores density already.

    Returns
    -------
    w : 1D array
        Gathered quantity at particle positions (same physical units as particle weights).
    all_gathered : bool
        Whether all particles were within bounds and gathered successfully.
    """
    if p_shape == "linear":
        return inverse_deposit_3d_distribution_linear(
            z, x, y,
            z_min, r_min, nz, nr, dz, dr,
            deposition_array,
            use_ruyten, r_min_deposit,
            input_is_cell_integrated
        )
    elif p_shape == "cubic":
        return inverse_deposit_3d_distribution_cubic(
            z, x, y,
            z_min, r_min, nz, nr, dz, dr,
            deposition_array,
            use_ruyten, r_min_deposit,
            input_is_cell_integrated
        )
    else:
        raise ValueError("p_shape must be 'linear' or 'cubic'.")


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
    input_is_cell_integrated=True,
):
    """Gather (CIC/linear) from grid to particles. Returns (w, all_gathered)."""

    # Precompute 1/V_r for interior radial cells (j=0..nr-1)
    if input_is_cell_integrated:
        v_cell = _build_cyl_cell_volumes(nr, dr, dz)
        inv_v = np.empty(nr)
        for j in range(nr):
            inv_v[j] = 1.0 / v_cell[j]
    else:
        inv_v = np.empty(nr)
        for j in range(nr):
            inv_v[j] = 1.0

    # Optional Ruyten coefficients (same expressions as deposit_3d_distribution_linear)
    if use_ruyten:
        ruyten_coef = np.zeros(nr + 1)
        # NOTE: deposit assumes cell-centered r grid; dz appears in volume but cancels out
        r_grid = (np.arange(nr) + 0.5) * dr
        cell_volume = math.pi * dz * ((r_grid + 0.5 * dr) ** 2 - (r_grid - 0.5 * dr) ** 2)
        cell_volume_norm = cell_volume / (2.0 * math.pi * dr * dr * dz)
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

            # SAME indexing convention as deposit_3d_distribution_linear
            ir_cell = min(int(math.ceil(r_cell)) + 1, nr + 2)
            iz_cell = int(math.ceil(z_cell)) + 1

            # u_r relative to left neighbor gridpoint in r (mirrors deposit)
            if r_cell < 0.0:
                u_r = 1.0
            else:
                u_r = r_cell - int(math.ceil(r_cell)) + 1.0

            # u_z relative to left neighbor gridpoint in z (FIXED typo: must depend on z_cell)
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

            # Convert cell-integrated -> density by dividing by V_r
            # Interior radial index j corresponds to ir = 2+j in the array.
            # For a guard-including index irg, interior j = irg - 2.
            def val_density(izg, irg):
                j = irg - 2
                if j < 0:
                    j = 0
                elif j >= nr:
                    j = nr - 1
                return deposition_array[izg, irg] * inv_v[j]

            w_i = 0.0
            w_i += zsl_0 * rsl_0 * val_density(iz_cell + 0, ir_cell + 0)
            w_i += zsl_0 * rsl_1 * val_density(iz_cell + 0, ir_cell + 1)
            w_i += zsl_1 * rsl_0 * val_density(iz_cell + 1, ir_cell + 0)
            w_i += zsl_1 * rsl_1 * val_density(iz_cell + 1, ir_cell + 1)

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
    input_is_cell_integrated=True,
):
    """Gather (3rd-order/cubic B-spline) from grid to particles. Returns (w, all_gathered)."""

    # Precompute 1/V_r for interior radial cells (j=0..nr-1)
    if input_is_cell_integrated:
        v_cell = _build_cyl_cell_volumes(nr, dr, dz)
        inv_v = np.empty(nr)
        for j in range(nr):
            inv_v[j] = 1.0 / v_cell[j]
    else:
        inv_v = np.empty(nr)
        for j in range(nr):
            inv_v[j] = 1.0

    # Optional Ruyten coefficients (same expressions as deposit_3d_distribution_cubic)
    if use_ruyten:
        ruyten_coef = np.zeros(nr + 1)
        r_grid = (np.arange(nr) + 0.5) * dr
        cell_volume = math.pi * dz * ((r_grid + 0.5 * dr) ** 2 - (r_grid - 0.5 * dr) ** 2)
        cell_volume_norm = cell_volume / (2.0 * math.pi * dr * dr * dz)
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

            # SAME base indices as deposit_3d_distribution_cubic
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

            # Apply SAME boundary-folding logic as deposit (important!)
            # Below axis:
            if r_cell <= 0.0:
                rsc_3 += rsc_0
                rsc_2 += rsc_1
                rsc_0 = 0.0
                rsc_1 = 0.0
            elif r_cell <= 1.0:
                rsc_1 += rsc_0
                rsc_0 = 0.0

            # Below z_min:
            if z_cell <= 0.0:
                zsc_3 += zsc_0
                zsc_2 += zsc_1
                zsc_0 = 0.0
                zsc_1 = 0.0
            elif z_cell <= 1.0:
                zsc_1 += zsc_0
                zsc_0 = 0.0
            # Above z_max:
            elif z_cell > nz - 1:
                zsc_0 += zsc_3
                zsc_1 += zsc_2
                zsc_2 = 0.0
                zsc_3 = 0.0
            elif z_cell > nz - 2:
                zsc_2 += zsc_3
                zsc_3 = 0.0

            # Convert cell-integrated -> density by dividing by V_r
            def val_density(izg, irg):
                j = irg - 2
                if j < 0:
                    j = 0
                elif j >= nr:
                    j = nr - 1
                return deposition_array[izg, irg] * inv_v[j]

            # Gather: 4x4 stencil
            w_i = 0.0
            w_i += zsc_0 * (rsc_0 * val_density(iz_cell + 0, ir_cell + 0) +
                            rsc_1 * val_density(iz_cell + 0, ir_cell + 1) +
                            rsc_2 * val_density(iz_cell + 0, ir_cell + 2) +
                            rsc_3 * val_density(iz_cell + 0, ir_cell + 3))
            w_i += zsc_1 * (rsc_0 * val_density(iz_cell + 1, ir_cell + 0) +
                            rsc_1 * val_density(iz_cell + 1, ir_cell + 1) +
                            rsc_2 * val_density(iz_cell + 1, ir_cell + 2) +
                            rsc_3 * val_density(iz_cell + 1, ir_cell + 3))
            w_i += zsc_2 * (rsc_0 * val_density(iz_cell + 2, ir_cell + 0) +
                            rsc_1 * val_density(iz_cell + 2, ir_cell + 1) +
                            rsc_2 * val_density(iz_cell + 2, ir_cell + 2) +
                            rsc_3 * val_density(iz_cell + 2, ir_cell + 3))
            w_i += zsc_3 * (rsc_0 * val_density(iz_cell + 3, ir_cell + 0) +
                            rsc_1 * val_density(iz_cell + 3, ir_cell + 1) +
                            rsc_2 * val_density(iz_cell + 3, ir_cell + 2) +
                            rsc_3 * val_density(iz_cell + 3, ir_cell + 3))

            w_out[i] = w_i
        else:
            all_gathered = False
            w_out[i] = 0.0

    return w_out, all_gathered
