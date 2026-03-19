"""
This module implements the methods for calculating the plasma wakefields
using the 2D r-z reduced model from P. Baxevanis and G. Stupakov.

See https://journals.aps.org/prab/abstract/10.1103/PhysRevAccelBeams.21.071301
for the full details about this model.
"""

import numpy as np
import scipy.constants as ct
import aptools.plasma_accel.general_equations as ge

from .plasma_particles import (
    pp_initialize,
    pp_sort,
    pp_gather_laser_sources,
    pp_gather_bunch_sources,
    pp_calculate_fields,
    pp_calculate_psi_at_grid,
    pp_calculate_b_theta_at_grid,
    pp_calculate_weights,
    pp_deposit_rho,
    pp_deposit_chi,
    pp_store_current_step,
    pp_evolve,
    pp_get_history,
)
from .utils import longitudinal_gradient, longitudinal_backward_gradient, radial_gradient

from wake_t.utilities.numba import njit_serial

from .plasma_particle_container import PlasmaParticleContainer


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


@njit_serial
def evolve_window_inplace(
    pp_serialized_list,
    n_xi,
    n_r,
    dxi,
    dr,
    r_fld,
    has_laser_source,
    laser_a2,
    nabla_a2,
    has_beam_source,
    bunch_source_arrays,
    bunch_source_xi_indices,
    bunch_source_metadata,
    max_gamma,
    psi,
    B_t,
    shape,
    calculate_rho,
    rho,
    rho_e,
    rho_i,
    chi,
    store_plasma_history,
    particle_diags,
    start_slice_i,  # inclusive
    stop_slice_i,  # inclusive (<= start_slice_i)
):
    """
    Resume from an existing pp_serialized_list and evolve only a small window.
    Mutates pp_serialized_list arrays in-place.
    """
    ions_computed = False
    pp_species_list = [
        PlasmaParticleContainer(species) for species in pp_serialized_list
    ]

    for slice_i in range(start_slice_i, stop_slice_i - 1, -1):
        pp_sort(pp_species_list)

        if has_laser_source:
            pp_gather_laser_sources(
                pp_species_list,
                laser_a2[slice_i + 2],
                nabla_a2[slice_i + 2],
                r_fld[0],
                r_fld[-1],
                dr,
            )

        if has_beam_source:
            pp_gather_bunch_sources(
                pp_species_list,
                bunch_source_arrays,
                bunch_source_xi_indices,
                bunch_source_metadata,
                slice_i,
            )

        pp_calculate_fields(pp_species_list, ions_computed, max_gamma)

        pp_calculate_psi_at_grid(pp_species_list, r_fld, psi[slice_i + 2, 2:-2])
        pp_calculate_b_theta_at_grid(pp_species_list, r_fld, B_t[slice_i + 2, 2:-2])

        if calculate_rho:
            pp_deposit_rho(
                pp_species_list,
                ions_computed,
                shape,
                rho[slice_i + 2],
                rho_e[slice_i + 2],
                rho_i[slice_i + 2],
                r_fld,
                n_r,
                dr,
            )
        elif "w" in particle_diags:
            pp_calculate_weights(pp_species_list, ions_computed)

        if has_laser_source:
            pp_deposit_chi(pp_species_list, shape, chi[slice_i + 2], r_fld, n_r, dr)

        ions_computed = True

        if store_plasma_history:
            pp_store_current_step(pp_species_list, particle_diags)

        if slice_i > 0:
            pp_evolve(pp_species_list, dxi)


@njit_serial
def evolve_one_step(
    pp_serialized_list,
    n_xi,
    n_r,
    dxi,
    dr,
    r_fld,
    has_laser_source,
    laser_a2,
    nabla_a2,
    has_beam_source,
    bunch_source_arrays,
    bunch_source_xi_indices,
    bunch_source_metadata,
    max_gamma,
    psi,
    B_t,
    shape,
    calculate_rho,
    rho,
    rho_e,
    rho_i,
    chi,
    store_plasma_history,
    particle_diags,
):
    """
    Compute the wakefield by evolving the plasma over all zeta slices.
    For performance reasons, this is done in a single JIT-compiled
    function to minimize the number of Python-to-Numba function calls.

    pp_serialized_list is passed in as a Tuple[Tuple[np.ndarray]]
    instead of a class so that Numba can cache the JIT compiled function.

    See calculate_wakefields() for parameters.
    """
    ions_computed = False
    pp_species_list = [
        PlasmaParticleContainer(species) for species in pp_serialized_list
    ]

    # Evolve plasma from right to left and calculate psi, b_t_bar, rho and
    # chi on a grid.
    for step in range(n_xi):
        slice_i = n_xi - step - 1
        pp_sort(pp_species_list)

        if has_laser_source:
            pp_gather_laser_sources(
                pp_species_list,
                laser_a2[slice_i + 2],
                nabla_a2[slice_i + 2],
                r_fld[0],
                r_fld[-1],
                dr,
            )
        if has_beam_source:
            pp_gather_bunch_sources(
                pp_species_list,
                bunch_source_arrays,
                bunch_source_xi_indices,
                bunch_source_metadata,
                slice_i,
            )

        pp_calculate_fields(pp_species_list, ions_computed, max_gamma)

        pp_calculate_psi_at_grid(pp_species_list, r_fld, psi[slice_i + 2, 2:-2])

        pp_calculate_b_theta_at_grid(pp_species_list, r_fld, B_t[slice_i + 2, 2:-2])

        if calculate_rho:
            pp_deposit_rho(
                pp_species_list,
                ions_computed,
                shape,
                rho[slice_i + 2],
                rho_e[slice_i + 2],
                rho_i[slice_i + 2],
                r_fld,
                n_r,
                dr,
            )
        elif "w" in particle_diags:
            pp_calculate_weights(pp_species_list, ions_computed)
        if has_laser_source:
            pp_deposit_chi(pp_species_list, shape, chi[slice_i + 2], r_fld, n_r, dr)

        ions_computed = True

        if store_plasma_history:
            pp_store_current_step(pp_species_list, particle_diags)

        if slice_i > 0:
            pp_evolve(pp_species_list, dxi)


def calculate_wakefields(
    laser_a2,
    r_max,
    xi_min,
    xi_max,
    n_r,
    n_xi,
    ppc,
    n_p,
    r_max_plasma=None,
    radial_density=None,
    p_shape="cubic",
    max_gamma=10.0,
    plasma_pusher="ab2",
    ion_motion=False,
    ion_mass=ct.m_p,
    free_electrons_per_ion=1,
    bunch_source_arrays=[],
    bunch_source_xi_indices=[],
    bunch_source_metadata=[],
    store_plasma_history=False,
    calculate_rho=True,
    particle_diags=[],
    fld_arrays=[],
):
    """
    Calculate the plasma wakefields generated by the given laser pulse and
    electron beam in the specified grid points.

    Parameters
    ----------
    laser_a2 : ndarray
        A (nz x nr) array containing the square of the laser envelope.
    r_max : float
        Maximum radial position up to which plasma wakefield will be
        calculated.
    xi_min : float
        Minimum longitudinal (speed of light frame) position up to which
        plasma wakefield will be calculated.
    xi_max : float
        Maximum longitudinal (speed of light frame) position up to which
        plasma wakefield will be calculated.
    n_r : int
        Number of grid elements along r in which to calculate the wakefields.
    n_xi : int
        Number of grid elements along xi in which to calculate the wakefields.
    ppc : array_like
        see Quasistatic2DWakefieldIons.
    n_p : float
        On-axis plasma density in units of m^{-3}.
    r_max_plasma : float
        Maximum radial extension of the plasma column. If `None`, the plasma
        extends up to the `r_max` boundary of the simulation box.
    radial_density : callable
        Function defining the radial density profile.
    p_shape : str
        Particle shape to be used for the beam charge deposition. Possible
        values are 'linear' or 'cubic'.
    max_gamma : float
        Plasma particles whose `gamma` exceeds `max_gamma` are considered to
        violate the quasistatic condition and are put at rest (i.e.,
        `gamma=1.`, `pr=pz=0.`).
    plasma_pusher : str
        Numerical pusher for the plasma particles. Possible values are `'ab2'`.
    ion_motion : bool, optional
        Whether to allow the plasma ions to move. By default, False.
    ion_mass : float, optional
        Mass of the plasma ions. By default, the mass of a proton.
    free_electrons_per_ion : int, optional
        Number of free electrons per ion. The ion charge is adjusted
        accordingly to maintain a quasi-neutral plasma (i.e.,
        ion charge = e * free_electrons_per_ion). By default, 1.
    bunch_source_arrays : list, optional
        List containing the array from which the bunch source terms (the
        azimuthal magnetic field) will be gathered. It can be a single
        array for the whole domain, or one array per bunch when using
        adaptive grids.
    bunch_source_xi_indices : list, optional
        List containing 1d arrays that with the indices of the longitudinal
        plasma slices that can gather from them. This is needed because the
        adaptive grids might not extend the whole longitudinal domain of the
        plasma, so the plasma slices should only try to gather the source terms
        if they are available at the current slice.
    bunch_source_metadata : list, optional
        Metadata of each bunch source array.
    store_plasma_history : bool, optional
        Whether to store the plasma particle evolution. This might be needed
        for diagnostics or the use of adaptive grids. By default, False.
    calculate_rho : bool, optional
        Whether to deposit the plasma density. This might be needed for
        diagnostics. By default, False.
    particle_diags : list, optional
        List of particle quantities to save to diagnostics.
    fld_arrays : list, optional
        List of all the fields.
    """
    rho, rho_e, rho_i, chi, E_r, E_z, B_t, xi_fld, r_fld = fld_arrays

    # Convert to normalized units.
    s_d = ge.plasma_skin_depth(n_p * 1e-6)
    r_max = r_max / s_d
    xi_min = xi_min / s_d
    xi_max = xi_max / s_d
    dr = r_max / n_r
    dxi = (xi_max - xi_min) / (n_xi - 1)
    ppc = ppc.copy()
    ppc[:, 0] /= s_d
    r_max_plasma = r_max_plasma / s_d

    def radial_density_normalized(r):
        return radial_density(r * s_d) / n_p

    # Field node coordinates.
    r_fld = r_fld / s_d
    xi_fld = xi_fld / s_d

    # Initialize field arrays, including guard cells.
    nabla_a2 = np.zeros((n_xi + 4, n_r + 4))
    psi = np.zeros((n_xi + 4, n_r + 4))

    # Laser source.
    has_laser_source = laser_a2 is not None
    if has_laser_source:
        radial_gradient(laser_a2[2:-2, 2:-2], dr, nabla_a2[2:-2, 2:-2])
    else:
        # need to set the dtype for JIT
        laser_a2 = np.zeros((0, 0))
        nabla_a2 = np.zeros((0, 0))

    has_beam_source = len(bunch_source_arrays) > 0
    if not has_beam_source:
        # need to set the dtype for JIT
        bunch_source_arrays.append(np.zeros((0, 0)))
        bunch_source_xi_indices.append(np.zeros(0, dtype=np.int64))
        bunch_source_metadata.append(np.zeros(0))

    if len(particle_diags) == 0:
        # need to set the type for JIT
        particle_diags = ["none"]

    # Calculate plasma response (including density, susceptibility, potential
    # and magnetic field)

    # Initialize plasma particles.
    # Set parameters for electron and ion species in normalized units
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

    species_list = pp_initialize(
        init_list,
        n_xi,
        ppc,
        dr,
        radial_density_normalized,
        ion_motion,
        store_plasma_history,
        plasma_pusher,
    )

    evolve_one_step(
        tuple(s.serialize() for s in species_list),
        n_xi,
        n_r,
        dxi,
        dr,
        r_fld,
        has_laser_source,
        laser_a2,
        nabla_a2,
        has_beam_source,
        tuple(bunch_source_arrays),
        tuple(bunch_source_xi_indices),
        tuple(bunch_source_metadata),
        max_gamma,
        psi,
        B_t,
        p_shape,
        calculate_rho,
        rho,
        rho_e,
        rho_i,
        chi,
        store_plasma_history,
        tuple(particle_diags),
    )

    # Calculate derived fields (E_z, W_r, and E_r).
    E_0 = ge.plasma_cold_non_relativisct_wave_breaking_field(n_p * 1e-6)
    #longitudinal_gradient(psi[2:-2, 2:-2], dxi, E_z[2:-2, 2:-2])
    longitudinal_backward_gradient(psi[2:-2, 2:-2], dxi, E_z[2:-2, 2:-2])
    radial_gradient(psi[2:-2, 2:-2], dr, E_r[2:-2, 2:-2])
    E_r -= B_t
    E_z *= -E_0
    E_r *= -E_0
    # B_t[:] = (b_t_bar + b_t_beam) * E_0 / ct.c
    B_t *= E_0 / ct.c
    return pp_get_history(species_list, store_plasma_history)


def calculate_wakefields_ez_km1_from_cache(
    pp_state_kp1,  # cached plasma state "after slice k+1" (ready for k)
    k,  # we are varying q_bunch at slice k, want Ez at k-1
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
):
    """
    Starting from cached plasma state at k+1, evolve only slices k..k-2
    and return Ez(k-1, r) (including guard r cells, consistent with your arrays).
    """
    rho, rho_e, rho_i, chi, E_r, E_z, B_t, xi_fld, r_fld = fld_arrays

    # normalize
    s_d = ge.plasma_skin_depth(n_p * 1e-6)
    r_max_n = r_max / s_d
    xi_min_n = xi_min / s_d
    xi_max_n = xi_max / s_d
    dr = r_max_n / n_r
    dxi = (xi_max_n - xi_min_n) / (n_xi - 1)
    r_fld_n = r_fld / s_d

    # minimal arrays
    nabla_a2 = np.zeros((n_xi + 4, n_r + 4))
    psi = np.zeros((n_xi + 4, n_r + 4))
    B_t[:] = 0.0
    chi[:] = 0.0

    has_laser_source = laser_a2 is not None
    if has_laser_source:
        radial_gradient(laser_a2[2:-2, 2:-2], dr, nabla_a2[2:-2, 2:-2])
    else:
        laser_a2 = np.zeros((0, 0))
        nabla_a2 = np.zeros((0, 0))

    # scratch state (don’t mutate cache)
    pp_trial = deepcopy_pp_state(pp_state_kp1)

    # evolve k, k-1, k-2 (need k>=2)
    evolve_window_inplace(
        pp_trial,
        n_xi,
        n_r,
        dxi,
        dr,
        r_fld_n,
        has_laser_source,
        laser_a2,
        nabla_a2,
        True,
        tuple(bunch_source_arrays),
        tuple(bunch_source_xi_indices),
        tuple(bunch_source_metadata),
        max_gamma,
        psi,
        B_t,
        p_shape,
        False,  # calculate_rho off
        rho,
        rho_e,
        rho_i,
        chi,
        False,  # no history
        ("none",),
        k,
        k - 2,
    )

    # compute Ez(k-1,r) using centered stencil in psi
    E_0 = ge.plasma_cold_non_relativisct_wave_breaking_field(n_p * 1e-6)
    psi_k = psi[2 + k, :]  # includes guard r
    psi_km2 = psi[2 + k - 2, :]  # includes guard r
    #Ez_r_km1 = -(psi_k - psi_km2) / (2.0 * dxi) * E_0
    psi_km1 = psi[2 + k - 1, :]  # includes guard r
    Ez_r_km1 = -(psi_km1 - psi_km2) / (1.0 * dxi) * E_0

    # psi_km1 = psi[2 + k - 1, :]  # includes guard r

    # print("psi(k) axis/mean/min/max:", psi_k[0], psi_k.mean(), psi_k.min(), psi_k.max())
    # print("psi(km1)   axis/mean/min/max:", psi_km1[0], psi_km1.mean(), psi_km1.min(), psi_km1.max())
    # print("psi(km2)   axis/mean/min/max:", psi_km2[0], psi_km2.mean(), psi_km2.min(), psi_km2.max())

    return Ez_r_km1


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
):
    rho, rho_e, rho_i, chi, E_r, E_z, B_t, xi_fld, r_fld = fld_arrays

    s_d = ge.plasma_skin_depth(n_p * 1e-6)
    r_max_n = r_max / s_d
    xi_min_n = xi_min / s_d
    xi_max_n = xi_max / s_d
    dr = r_max_n / n_r
    dxi = (xi_max_n - xi_min_n) / (n_xi - 1)

    ppc_n = ppc.copy()
    ppc_n[:, 0] /= s_d

    def radial_density_normalized(r):
        return radial_density(r * s_d) / n_p

    r_fld_n = r_fld / s_d

    nabla_a2 = np.zeros((n_xi + 4, n_r + 4))
    psi = np.zeros((n_xi + 4, n_r + 4))
    B_t[:] = 0.0
    chi[:] = 0.0

    has_laser_source = laser_a2 is not None
    if has_laser_source:
        radial_gradient(laser_a2[2:-2, 2:-2], dr, nabla_a2[2:-2, 2:-2])
    else:
        laser_a2 = np.zeros((0, 0))
        nabla_a2 = np.zeros((0, 0))

    init_list = [
        {
            "charge": free_electrons_per_ion,
            "mass": free_electrons_per_ion,
            "is_ion": False,
        },
        {"charge": -free_electrons_per_ion, "mass": ion_mass / ct.m_e, "is_ion": True},
    ]
    species_list = pp_initialize(
        init_list,
        n_xi,
        ppc_n,
        dr,
        radial_density_normalized,
        ion_motion,
        False,  # store_history OFF (critical for cheap copies)
        plasma_pusher,
    )
    pp_cache = tuple(s.serialize() for s in species_list)

    # evolve tail -> k_tail+1 to make cache "ready for k_tail"
    evolve_window_inplace(
        pp_cache,
        n_xi,
        n_r,
        dxi,
        dr,
        r_fld_n,
        has_laser_source,
        laser_a2,
        nabla_a2,
        True,
        tuple(bunch_source_arrays),
        tuple(bunch_source_xi_indices),
        tuple(bunch_source_metadata),
        max_gamma,
        psi,
        B_t,
        p_shape,
        False,
        rho,
        rho_e,
        rho_i,
        chi,
        False,
        ("none",),
        n_xi - 1,
        k_tail + 1,
    )
    return pp_cache


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
):
    rho, rho_e, rho_i, chi, E_r, E_z, B_t, xi_fld, r_fld = fld_arrays

    s_d = ge.plasma_skin_depth(n_p * 1e-6)
    r_max_n = r_max / s_d
    xi_min_n = xi_min / s_d
    xi_max_n = xi_max / s_d
    dr = r_max_n / n_r
    dxi = (xi_max_n - xi_min_n) / (n_xi - 1)
    r_fld_n = r_fld / s_d

    nabla_a2 = np.zeros((n_xi + 4, n_r + 4))
    psi = np.zeros((n_xi + 4, n_r + 4))
    B_t[:] = 0.0
    chi[:] = 0.0

    has_laser_source = laser_a2 is not None
    if has_laser_source:
        radial_gradient(laser_a2[2:-2, 2:-2], dr, nabla_a2[2:-2, 2:-2])
    else:
        laser_a2 = np.zeros((0, 0))
        nabla_a2 = np.zeros((0, 0))

    # evolve exactly slice k
    evolve_window_inplace(
        pp_cache,
        n_xi,
        n_r,
        dxi,
        dr,
        r_fld_n,
        has_laser_source,
        laser_a2,
        nabla_a2,
        True,
        tuple(bunch_source_arrays),
        tuple(bunch_source_xi_indices),
        tuple(bunch_source_metadata),
        max_gamma,
        psi,
        B_t,
        p_shape,
        False,
        rho,
        rho_e,
        rho_i,
        chi,
        False,
        ("none",),
        k,
        k,
    )
