from typing import Optional, Callable, List, Union, Dict
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike
import scipy.constants as ct
import aptools.plasma_accel.general_equations as ge

from .solver import calculate_wakefields
from .b_theta_bunch import calculate_bunch_source, deposit_bunch_charge
from .adaptive_grid import AdaptiveGrid
from .utils import calculate_laser_a2
from wake_t.fields.rz_wakefield import RZWakefield
from wake_t.physics_models.laser.laser_pulse import LaserPulse
from wake_t.particles.particle_bunch import ParticleBunch
from wake_t.particles.interpolation import gather_main_fields_cyl_linear

from wake_t.particles.inverse_deposition import inverse_deposit_3d_distribution
from wake_t.particles.deposition import deposit_3d_distribution

class Quasistatic2DWakefieldIon(RZWakefield):
    """
    This class calculates the plasma wakefields using the gridless
    quasi-static model in r-z geometry originally developed by P. Baxevanis
    and G. Stupakov [1]_.

    The model implemented here includes additional features with respect to the
    original version in [1]_. Among them is the support for laser drivers,
    particle beams (instead of an analytic charge distribution), non-uniform
    and finite plasma density profiles, as well as an Adams-Bashforth pusher
    for the plasma particles (in addition the original Runke-Kutta pusher).

    As a kinetic quasi-static model in r-z geometry, it computes the plasma
    response by evolving a 1D radial plasma column from the front to the back
    of the simulation domain. A special feature of this model is that it does
    not need a grid in order to compute this evolution, and it allows the
    fields ``E`` and ``B`` to be calculated at any radial position in the
    plasma column.

    In the Wake-T implementation, a grid is used only in order to be able
    to easily interpolate the fields to the beam particles. After evolving the
    plasma column, ``E`` and ``B`` are calculated at the locations of the grid
    nodes. Similarly, the charge density ``rho`` and susceptibility ``chi``
    of the plasma are computed by depositing the charge of the plasma
    particles on the same grid. This useful for diagnostics and for evolving
    a laser pulse.

    Parameters
    ----------
    density_function : callable
        Function that returns the density value at the given position (z, r).
        This parameter is given by the ``PlasmaStage`` and does not need
        to be specified by the user.
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
        Number of grid elements along `r` to calculate the wakefields.
    n_xi : int
        Number of grid elements along `xi` to calculate the wakefields.
    ppc : array_like, optional
        Number of plasma particles per radial cell. It can be a single number
        (e.g., ``ppc=2``) if the plasma should have the same number of
        particles per cell everywhere. Alternatively, a different number
        of particles per cell at different radial locations can also be
        specified. This can be useful, for example, when using adaptive grids
        with very narrow beams that might require more plasma particles close
        to the axis. To achieve this, an array-like structure should be given
        where each item contains two values: the number of particles per cell
        and the radius up to which this number should be used. For example
        to have 8 ppc up to a radius of 100µm and 2 ppc for higher radii up to
        500µm ``ppc=[[100e-6, 8], [500e-6, 2]]``. When using this step option
        for ``ppc`` the ``r_max_plasma`` argument is ignored. By default
        ``ppc=2``.
    dz_fields : float, optional
        Determines how often the plasma wakefields should be updated.
        For example, if ``dz_fields=10e-6``, the plasma wakefields are
        only updated every time the simulation window advances by
        10 micron. By default ``dz_fields=xi_max-xi_min``, i.e., the
        length the simulation box.
    r_max_plasma : float, optional
        Maximum radial extension of the plasma column. If ``None``, the
        plasma extends up to the ``r_max`` boundary of the simulation box.
    p_shape : str, optional
        Particle shape to be used for the beam charge deposition. Possible
        values are ``'linear'`` or ``'cubic'`` (default).
    max_gamma : float, optional
        Plasma particles whose ``gamma`` exceeds ``max_gamma`` are
        considered to violate the quasistatic condition and are put at
        rest (i.e., ``gamma=1.``, ``pr=pz=0.``). By default
        ``max_gamma=10``.
    plasma_pusher : str, optional
        The pusher used to evolve the plasma particles. Possible values
        are ``'ab2'`` (Adams-Bashforth 2nd order).
    ion_motion : bool, optional
        Whether to allow the plasma ions to move. By default, False.
    ion_mass : float, optional
        Mass of the plasma ions. By default, the mass of a proton.
    free_electrons_per_ion : int, optional
        Number of free electrons per ion. The ion charge is adjusted
        accordingly to maintain a quasi-neutral plasma (i.e.,
        ion charge = e * free_electrons_per_ion). By default, 1.
    laser : LaserPulse, optional
        Laser driver of the plasma stage.
    laser_evolution : bool, optional
        If True (default), the laser pulse is evolved
        using a laser envelope model. If ``False``, the pulse envelope
        stays unchanged throughout the computation.
    laser_envelope_substeps : int, optional
        Number of substeps of the laser envelope solver per ``dz_fields``.
        The time step of the envelope solver is therefore
        ``dz_fields / c / laser_envelope_substeps``.
    laser_envelope_nxi, laser_envelope_nr : int, optional
        If given, the laser envelope will run in a grid of size
        (``laser_envelope_nxi``, ``laser_envelope_nr``) instead
        of (``n_xi``, ``n_r``). This allows the laser to run in a finer (or
        coarser) grid than the plasma wake. It is not necessary to specify
        both parameters. If one of them is not given, the resolution of
        the plasma grid with be used for that direction.
    laser_envelope_use_phase : bool, optional
        Determines whether to take into account the terms related to the
        longitudinal derivative of the complex phase in the envelope
        solver.
    field_diags : list, optional
        List of fields to save to openpmd diagnostics. By default ['rho', 'E',
        'B', 'a_mod', 'a_phase'].
    field_diags : list, optional
        List of particle quantities to save to openpmd diagnostics. By default
        [].
    use_adaptive_grids : bool, optional
        Whether to use adaptive grids for each particle bunch, instead of the
        general (n_xi x n_r) grid.
    adaptive_grid_nr : int or list of int, optional
        Radial resolution of the adaptive grids. If only one value is given,
        the same resolution will be used for the adaptive grids of all bunches.
        Otherwise, a list of values can be given (one per bunch and in the same
        order as the list of bunches given to the `track` method). If the
        value is `None`, no adaptive grid will be used for the corresponding
        bunch, which will instead use the base grid.
    adaptive_grid_r_max : float or list of float, optional
        Specify a fixed radial extent for the adaptive grids. If not given,
        the radial extent of the grids is continuously adapted with the
        transverse size of the bunches. If only one value is given,
        the same extent will be used for all adaptive grids.
        Otherwise, a list of values can be given (one per bunch and in the same
        order as the list of bunches given to the `track` method). The
        individual values can be `float` or `None` (in which case, no fixed
        radial extent is used for the corresponding grid).
    adaptive_grid_r_lim : float or list of float, optional
        Specify a limit to the radial extent of the adaptive grids. If not
        given, the radial extent of the grids is continuously adapted to fit
        the whole transverse size of the bunches. If only one value is given,
        the same limit will be used for all adaptive grids.
        Otherwise, a list of values can be given (one per bunch and in the same
        order as the list of bunches given to the `track` method). The
        individual values can be `float` or `None` (in which case, no radial
        limit is used for the corresponding grid). Bunch particles that escape
        the grid transversely with deposit to and gather from the base grid
        (if they haven't escaped from it too).
    adaptive_grid_diags : list, optional
        List of fields from the adaptive grids to save to openpmd diagnostics.
        By default ['E', 'B'].

    References
    ----------
    .. [1] P. Baxevanis and G. Stupakov, "Novel fast simulation technique
        for axisymmetric plasma wakefield acceleration configurations in
        the blowout regime," Phys. Rev. Accel. Beams 21, 071301 (2018),
        https://link.aps.org/doi/10.1103/PhysRevAccelBeams.21.071301

    """

    def __init__(
        self,
        density_function: Callable[[float, float], float],
        r_max: float,
        xi_min: float,
        xi_max: float,
        n_r: int,
        n_xi: int,
        ppc: Optional[ArrayLike] = 2,
        dz_fields: Optional[float] = None,
        r_max_plasma: Optional[float] = None,
        parabolic_coefficient: Optional[float] = None,
        p_shape: Optional[str] = "cubic",
        max_gamma: Optional[float] = 10,
        plasma_pusher: Optional[str] = "ab2",
        ion_motion: Optional[bool] = False,
        ion_mass: Optional[float] = ct.m_p,
        free_electrons_per_ion: Optional[int] = 1,
        laser: Optional[LaserPulse] = None,
        laser_evolution: Optional[bool] = True,
        laser_envelope_substeps: Optional[int] = 1,
        laser_envelope_nxi: Optional[int] = None,
        laser_envelope_nr: Optional[int] = None,
        laser_envelope_use_phase: Optional[bool] = True,
        field_diags: Optional[List[str]] = [
            "rho",
            "E",
            "B",
            "a",
        ],
        particle_diags: Optional[List[str]] = [],
        use_adaptive_grids: Optional[bool] = False,
        adaptive_grid_nr: Optional[Union[int, List[int]]] = 16,
        adaptive_grid_r_max: Optional[Union[float, List[float]]] = None,
        adaptive_grid_r_lim: Optional[Union[float, List[float]]] = None,
        adaptive_grid_diags: Optional[List[str]] = ["E", "B"],
    ) -> None:
        # Checks for backward compatibility.
        if plasma_pusher not in ["ab2"]:
            if plasma_pusher in ["rk4", "ab5"]:
                plasma_pusher = "ab2"
                warn(
                    "Plasma pusher {plasma_pusher} has been deprecated. "
                    "Using 'ab2' pusher instead.",
                    DeprecationWarning,
                )
            else:
                raise ValueError(
                    "Plasma pusher {plasma_pusher} not recognized. "
                    "Possible values are ['ab2']."
                )
        if parabolic_coefficient is not None:
            warn(
                "The 'parabolic_coefficient' parameter has been deprecated. "
                "It will still work in order to maintain backward "
                "compatibility, but the "
                "new recommended approach is to provide a density function "
                "that takes `z` and `r` as input parameters.",
                DeprecationWarning,
            )
            parabolic_coefficient = self._get_parabolic_coefficient_fn(
                parabolic_coefficient
            )
        if r_max_plasma is None:
            r_max_plasma = r_max - r_max / n_r / 2
        self.r_max_plasma = r_max_plasma
        self.ppc = np.array(ppc)
        self.p_shape = p_shape
        self.max_gamma = max_gamma
        self.plasma_pusher = plasma_pusher
        self.ion_motion = ion_motion
        self.ion_mass = ion_mass
        self.free_electrons_per_ion = free_electrons_per_ion
        self.use_adaptive_grids = use_adaptive_grids
        self.adaptive_grid_nr = adaptive_grid_nr
        self.adaptive_grid_r_max = adaptive_grid_r_max
        self.adaptive_grid_r_lim = adaptive_grid_r_lim
        self.adaptive_grid_diags = adaptive_grid_diags
        self.bunch_grids: Dict[str, AdaptiveGrid] = {}
        self._t_reset_bunch_arrays = -1.0
        if len(self.ppc.shape) in [0, 1]:
            self.ppc = np.array([[self.r_max_plasma, self.ppc.flatten()[0]]])
        self.parabolic_coefficient = parabolic_coefficient

        super().__init__(
            density_function=density_function,
            r_max=r_max,
            xi_min=xi_min,
            xi_max=xi_max,
            n_r=n_r,
            n_xi=n_xi,
            dz_fields=dz_fields,
            species_rho_diags=True,
            laser=laser,
            laser_evolution=laser_evolution,
            laser_envelope_substeps=laser_envelope_substeps,
            laser_envelope_nxi=laser_envelope_nxi,
            laser_envelope_nr=laser_envelope_nr,
            laser_envelope_use_phase=laser_envelope_use_phase,
            field_diags=field_diags,
            particle_diags=particle_diags,
            model_name="quasistatic_2d_ion",
        )


        # --- one-time init & one-time application flags ---
        #self._did_init_ic = False              # init hook has been executed?
        #self._apply_beamloading_this_solve = True  # apply beam-loading only on first solve?
        self._initial_condition_done = False



    def _initialize_properties(self, bunches):
        super()._initialize_properties(bunches)
        # Add bunch source array (needed if not using adaptive grids).
        self.b_t_bunch = np.zeros((self.n_xi + 4, self.n_r + 4))
        self.q_bunch = np.zeros((self.n_xi + 4, self.n_r + 4))
        self.laser_a2 = np.zeros((self.n_xi + 4, self.n_r + 4))
        self.fld_arrays = [
            self.rho,
            self.rho_e,
            self.rho_i,
            self.chi,
            self.e_r,
            self.e_z,
            self.b_t,
            self.xi_fld,
            self.r_fld,
        ]


    def _apply_beamloading_shaper_on_qbunch(
        self,
        xi_min: float,
        xi_max: float,
        scale: Union[float, np.ndarray],
        smooth: int = 0,
    ):
        """
        Multiply q_bunch(ξ,r) by a scale factor in a given ξ interval.

        Parameters
        ----------
        xi_min, xi_max : float
        Interval in xi (same units as self.xi_fld, i.e. meters in Wake-T).
        This selects slices whose cell centers fall in [xi_min, xi_max].
        scale : float or 1D array
        If float: uniform scaling in the interval.
        If array: per-slice scaling for the selected slices (length = n_sel).
        smooth : int
        If >0, apply a simple moving-average smoothing to the *scale* along xi
        before multiplying. (Helps avoid sharp Ez spikes.)
        """
        # Interior xi cell centers (exclude guards)
        xi_centers = self.xi_fld[2:-2]          # shape (n_xi,)
        sel = (xi_centers >= xi_min) & (xi_centers <= xi_max)
        idx = np.where(sel)[0]                  # indices in 0..n_xi-1

        if idx.size == 0:
            return

        if np.isscalar(scale):
            s = np.full(idx.size, float(scale), dtype=float)
        else:
            s = np.asarray(scale, dtype=float)
            if s.size != idx.size:
                raise ValueError(
                    f"scale has length {s.size}, but selected {idx.size} xi slices."
                )

        if smooth and smooth > 0 and s.size > 2:
            # moving average on s
            k = int(smooth)
            kernel = np.ones(2 * k + 1, dtype=float)
            kernel /= kernel.sum()
            s_pad = np.pad(s, (k, k), mode="edge")
            s = np.convolve(s_pad, kernel, mode="valid")

        # Apply scaling on q_bunch for all r (including guard r cells if you want)
        # q_bunch is (n_xi+4, n_r+4). Interior r is 2:-2.
        for ii, si in zip(idx, s):
            self.q_bunch[2 + ii, 2:-2] *= si



#    def _init_initial_condition_once(self, bunches: List[ParticleBunch]):
#        """
#        Run exactly once, at the beginning of the first wake solve.
#
#        Use this to precompute any arrays for beam-loading shaping.
#        """
#        # Choose which bunch defines the beam-loading target (often the witness or driver).
#        # For now pick bunches[0]; change selection logic as you like.
#        bunch = bunches[0]
#
#
#        # store parameters for one-time shaper (or compute them)
#        self._beamload_xi_min = -64e-6
#        self._beamload_xi_max = -64e-6 + 1 * self.dxi
#        self._beamload_scale  = 1.0
#        self._beamload_smooth = 0
#
#
#        # Example: build q_scale_xi aligned to the base xi grid (interior indices).
#        # You must decide the desired profile; here is a placeholder.
#        n_xi = self.n_xi
#        q_scale_xi = np.ones(n_xi, dtype=np.float64)
#
#        # --- USER LOGIC GOES HERE ---
#        # e.g. q_scale_xi[k] = ...
#        # using self.xi_fld[2:-2] (interior xi nodes) if needed:
#        # xi_centers = self.xi_fld[2:-2]
#        # q_scale_xi = some_function(xi_centers, bunch, ...)
#        # --------------------------------
#
#        #self.q_scale_xi = q_scale_xi
#        #self.xi_min_shaper = self.xi_fld[2]   # consistent with interior indexing
#        #self.use_q_shaper = True              # we have shaper data now
#        print("hello")



    def _apply_beamloading_once_on_deposited_qbunch(self):
        # optional debug
        # print(self.xi_fld); print(self.dxi); print([np.max(self.q_bunch), np.min(self.q_bunch)])
    
        self._beamload_xi_min = -64e-6
        self._beamload_xi_max = -64e-6 + 5 * self.dxi
        self._beamload_scale  = 0.3 
        self._beamload_smooth = 0
        self._apply_beamloading_shaper_on_qbunch(
            xi_min=self._beamload_xi_min,
            xi_max=self._beamload_xi_max,
            scale=self._beamload_scale,
            smooth=self._beamload_smooth,
        )
    
        # optional debug
        # print([np.max(self.q_bunch), np.min(self.q_bunch)])




#    def _beamloading_initial_condition(self,
#                                   laser_a2,
#                                   radial_density,
#                                   store_plasma_history,
#                                   bunch_source_arrays,
#                                   bunch_source_xi_indices,
#                                   bunch_source_metadata,
#                                   bunches: List[ParticleBunch]):
#        """
#        Runs ONLY ONCE.
#        May call calculate_wakefields multiple times internally.
#        Leaves q_bunch / b_t_bunch / fld_arrays in final consistent state.
#        """
#
#
#        if len(bunch_source_arrays) == 0:
#            bunch_source_arrays.append(self.b_t_bunch)
#            bunch_source_xi_indices.append(np.arange(self.n_xi))
#    
#            s_d = ge.plasma_skin_depth(self.n_p * 1e-6)
#            bunch_source_metadata.append(
#                np.array([self.r_fld[0], self.r_fld[-1] + 2 * self.dr, self.dr]) / s_d
#            )
#
#
#        N_ITER = 1
#        
#        for _ in range(N_ITER):   # placeholder
#
#            # modify self.q_bunch
#            print(self.xi_fld)
#            print(self.dxi)
#            print([np.max(self.q_bunch),np.min(self.q_bunch)])
#    
#    
#            # ---- INSERT beam-loading shaper HERE ----
#            # Example: increase witness charge by 5% between xi=-200um and -100um
#            #self._apply_beamloading_once_on_deposited_qbunch()
#
#            #self._beamload_xi_min = -64e-6
#            #self._beamload_xi_max = -64e-6 + 5 * self.dxi
#            #self._beamload_scale  = 0.3 
#            #self._beamload_smooth = 0
#            #self._apply_beamloading_shaper_on_qbunch(
#            #    xi_min=self._beamload_xi_min,
#            #    xi_max=self._beamload_xi_max,
#            #    scale=self._beamload_scale,
#            #    smooth=self._beamload_smooth,
#            #)
#
#
#
#            self._apply_beamloading_shaper_on_qbunch(xi_min=-64e-6, xi_max=-64e-6+1*self.dxi,
#                                                      scale=0.1, smooth=0)
#            print([np.max(self.q_bunch),np.min(self.q_bunch)])
#
#
#            calculate_bunch_source(self.q_bunch, self.n_r, self.n_xi, self.b_t_bunch)
#    
##            bunch_source_arrays.append(self.b_t_bunch)
##            bunch_source_xi_indices.append(np.arange(self.n_xi))
##
##            s_d = ge.plasma_skin_depth(self.n_p * 1e-6)
##            bunch_source_metadata.append(
##                np.array(
##                    [
##                        self.r_fld[0],
##                        self.r_fld[-1] + 2 * self.dr,  # r of last guard cell.
##                        self.dr,
##                    ]
##                )
##                / s_d
##            )
#
#
#
#            # overwrite the existing base-grid slot (index 0)
#            bunch_source_arrays[0] = self.b_t_bunch
#
#
#            # Calculate rho only if requested in the diagnostics.
#            calculate_rho = any("rho" in diag for diag in self.field_diags)
#
#
#            self.pp = calculate_wakefields(
#                laser_a2,
#                self.r_max,
#                self.xi_min,
#                self.xi_max,
#                self.n_r,
#                self.n_xi,
#                self.ppc,
#                self.n_p,
#                r_max_plasma=self.r_max_plasma,
#                radial_density=radial_density,
#                p_shape=self.p_shape,
#                max_gamma=self.max_gamma,
#                plasma_pusher=self.plasma_pusher,
#                ion_motion=self.ion_motion,
#                ion_mass=self.ion_mass,
#                free_electrons_per_ion=self.free_electrons_per_ion,
#                fld_arrays=self.fld_arrays,
#                bunch_source_arrays=bunch_source_arrays,
#                bunch_source_xi_indices=bunch_source_xi_indices,
#                bunch_source_metadata=bunch_source_metadata,
#                store_plasma_history=False,
#                calculate_rho=calculate_rho,
#                particle_diags=[],
#            )
#
#
#            E_z = self.fld_arrays[5]
#            E_z_phys = E_z[2:-2, 2:-2]   # interior (no guard cells)
#            print(E_z_phys[:,0])
#
#
#
#
#
#
#
#            # Add bunch density to total density.
#            if calculate_rho:
#                rho_bunch = -self.q_bunch[2:-2, 2:-2] / (self.r_fld / s_d)
#                self.rho[2:-2, 2:-2] += rho_bunch
#    
#            # Calculate fields on adaptive grids.
#            if self.use_adaptive_grids:
#                for _, grid in self.bunch_grids.items():
#                    grid.calculate_fields(self.n_p, self.pp)
#
#
#
#        q_new, _ = inverse_deposit_3d_distribution(
#            bunches[0].xi, bunches[0].x, bunches[0].y,
#            self.xi_fld[0], self.r_fld[0],
#            self.n_xi, self.n_r, self.dxi, self.dr,
#            self.q_bunch,
#            p_shape="cubic",
#        )
#
#        bunches[0].w = np.abs(q_new/bunches[0].q_species)




    def _beamloading_initial_condition(
        self,
        laser_a2,
        radial_density,
        store_plasma_history,
        bunch_source_arrays,
        bunch_source_xi_indices,
        bunch_source_metadata,
        bunches: List[ParticleBunch],
    ):
        """
        Runs ONLY ONCE.
        Shapes q_bunch via 1D q_bunch_line so that bunch-weighted <Ez>(xi)
        becomes flat at the tail value.
        """
    
        # --- keep your existing bunch_source init ---
        if len(bunch_source_arrays) == 0:
            bunch_source_arrays.append(self.b_t_bunch)
            bunch_source_xi_indices.append(np.arange(self.n_xi))
    
            s_d = ge.plasma_skin_depth(self.n_p * 1e-6)
            bunch_source_metadata.append(
                np.array([self.r_fld[0], self.r_fld[-1] + 2 * self.dr, self.dr]) / s_d
            )
    
        # ----------------- helpers (local) -----------------
        #r_centers = self.r_fld[2:-2]                 # (n_r,)
        #xi_centers = self.xi_fld[2:-2]               # (n_xi,)
        r_centers = self.r_fld                 # (n_r,)
        xi_centers = self.xi_fld               # (n_xi,)
    
        # 1D line charge (C/m) from deposited q_bunch(ξ,r)
        # NOTE: q_bunch is charge per (xi,r) cell (C). For line-charge you need sum over r
        # and divide by dxi (and include 2πr Jacobian if your deposit is "volume-like").
        # If your deposit already accounts for cylindrical Jacobian, drop (2πr*dr).
        def q_bunch_line_from_qbunch(qb2d: np.ndarray) -> np.ndarray:
            qb = qb2d[2:-2, 2:-2]  # (n_xi,n_r)
            # conservative cylindrical line integration:
            # g(ξ) ≈ (1/dxi) ∫ 2π r (charge_density_like) dr
            # Here qb is "cell charge", so include 2πr and dr, then divide by dxi:
            return (np.sum(qb * (2.0 * np.pi * r_centers[None, :]) , axis=1) * self.dr) / self.dxi
    
        # bunch-weighted <Ez>(ξ) using deposited charge magnitude as weights in r
        def Ez_weighted_from_fields(Ez2d: np.ndarray, qb2d: np.ndarray) -> np.ndarray:
            Ez = Ez2d[2:-2, 2:-2]          # (n_xi,n_r)
            qb = qb2d[2:-2, 2:-2]
            w = np.abs(qb)
            wsum = np.sum(w, axis=1)
            out = np.zeros(Ez.shape[0], dtype=float)
            m = wsum > 0
            out[m] = np.sum(Ez[m, :] * w[m, :], axis=1) / wsum[m]
            return out
    
        # scale one xi slice k (in interior indexing 0..n_xi-1) to set a new line-charge g_new
        # keeps transverse shape fixed
        def set_slice_line_charge(qb2d: np.ndarray, k: int, g_new: float) -> np.ndarray:
            qb_new = qb2d.copy()
            g_old_all = q_bunch_line_from_qbunch(qb2d)
            g_old = g_old_all[k]
            if g_old == 0.0:
                return qb_new
            s = g_new / g_old
            qb_new[2 + k, 2:-2] *= s
            return qb_new
    
        def solve_with_qbunch(qb2d: np.ndarray):
            # update q_bunch and b_t_bunch
            self.q_bunch[:, :] = qb2d
            calculate_bunch_source(self.q_bunch, self.n_r, self.n_xi, self.b_t_bunch)
    
            # overwrite the base-grid slot (index 0) to use the updated b_t_bunch
            bunch_source_arrays[0] = self.b_t_bunch
    
            calculate_rho = any("rho" in diag for diag in self.field_diags)
    
            self.pp = calculate_wakefields(
                laser_a2,
                self.r_max,
                self.xi_min,
                self.xi_max,
                self.n_r,
                self.n_xi,
                self.ppc,
                self.n_p,
                r_max_plasma=self.r_max_plasma,
                radial_density=radial_density,
                p_shape=self.p_shape,
                max_gamma=self.max_gamma,
                plasma_pusher=self.plasma_pusher,
                ion_motion=self.ion_motion,
                ion_mass=self.ion_mass,
                free_electrons_per_ion=self.free_electrons_per_ion,
                fld_arrays=self.fld_arrays,
                bunch_source_arrays=bunch_source_arrays,
                bunch_source_xi_indices=bunch_source_xi_indices,
                bunch_source_metadata=bunch_source_metadata,
                store_plasma_history=False,      # IC stage only
                calculate_rho=calculate_rho,
                particle_diags=[],
            )
    
            Ez_w = Ez_weighted_from_fields(self.e_z, self.q_bunch)
            g_line = q_bunch_line_from_qbunch(self.q_bunch)
            return Ez_w, g_line
    
        # ----------------- SALAME-like shaping -----------------
    
        # starting point: current deposited qbunch from this timestep
        qb_current = self.q_bunch.copy()
    
        Ez_w0, g_line0 = solve_with_qbunch(qb_current)
    
        # define bunch-support indices (where there is charge)
        support = np.where(np.abs(g_line0) > 0.0)[0]
        if support.size < 2:
            # nothing to shape
            return
    
        k_tail = support[-1]
        Ez_target = Ez_w0[k_tail]   # tail value target (flat)
    
        # parameters (keep minimal)
        max_iter = 200
        tol = 1e-4
    
        # march from tail -> head (skip first support index because we use k-1 control)
        for k in range(k_tail, support[0], -1):
            # only shape inside the bunch-support region
            if np.abs(g_line0[k]) == 0.0:
                continue
    
            # bracket on g_k
            g_min = 0.0
            g_max = 5.0 * g_line0[k]   # start near original
    
            qb_min = set_slice_line_charge(qb_current, k, g_min)
            Ez_min, _ = solve_with_qbunch(qb_min)
    
            qb_max = set_slice_line_charge(qb_current, k, g_max)
            Ez_max, _ = solve_with_qbunch(qb_max)
    
            # expand g_max until we bracket target at control point k-1
            # (match magnitude; same logic as your script)
            while np.abs(Ez_max[k - 1]) > np.abs(Ez_target):
                g_max *= 5.0
                qb_max = set_slice_line_charge(qb_current, k, g_max)
                Ez_max, _ = solve_with_qbunch(qb_max)
                # safety
                if g_max == 0.0 or not np.isfinite(g_max):
                    break
    
            # regula-falsi style refinement
            qb_new = qb_current
            Ez_new = Ez_min
            for _ in range(max_iter):
                den = np.abs(Ez_max[k - 1] - Ez_min[k - 1])
                if den == 0.0:
                    break
    
                wg = np.abs(Ez_target - Ez_min[k - 1]) / den
                g_new = wg * g_max + (1.0 - wg) * g_min
    
                qb_try = set_slice_line_charge(qb_current, k, g_new)
                Ez_try, _ = solve_with_qbunch(qb_try)
    
                # update bracket
                if np.abs(Ez_try[k - 1]) > np.abs(Ez_target):
                    g_min, Ez_min = g_new, Ez_try
                else:
                    g_max, Ez_max = g_new, Ez_try
    
                rel = np.abs(Ez_try[k - 1] - Ez_target) / (np.abs(Ez_target) + 1e-300)
                qb_new, Ez_new = qb_try, Ez_try
                if rel < tol:
                    break
    
            # accept this slice update and move to next slice forward
            qb_current = qb_new
    
        # finalize q_bunch with shaped qb_current
        self.q_bunch[:, :] = qb_current
        calculate_bunch_source(self.q_bunch, self.n_r, self.n_xi, self.b_t_bunch)
        bunch_source_arrays[0] = self.b_t_bunch
    
        # map shaped deposited qbunch -> particle charges, update bunch weights (your existing code)
        q_new, _ = inverse_deposit_3d_distribution(
            bunches[0].xi, bunches[0].x, bunches[0].y,
            self.xi_fld[0], self.r_fld[0],
            self.n_xi, self.n_r, self.dxi, self.dr,
            self.q_bunch,
            p_shape="cubic",
        )
        bunches[0].w = np.abs(q_new / bunches[0].q_species)







    def _calculate_wakefield(self, bunches: List[ParticleBunch]):
        # --- initial-condition-only init ---
#        if not self._did_init_ic:
#            self._init_initial_condition_once(bunches)
#            self._did_init_ic = True


        radial_density = self._get_radial_density(self.t * ct.c)

        # Get square of laser envelope
        if self.laser is not None:
            calculate_laser_a2(self.laser.get_envelope(), self.laser_a2)
            # If linearly polarized, divide by 2 so that the ponderomotive
            # force on the plasma particles is correct.
            if self.laser.polarization == "linear":
                self.laser_a2 /= 2.0
            laser_a2 = self.laser_a2
        else:
            laser_a2 = None

        # Store plasma history if required by the diagnostics.
        store_plasma_history = len(self.particle_diags) > 0

        # Initialize empty lists with correct type so that numba can use
        # them even if there are no bunch sources.
        bunch_source_arrays = []
        bunch_source_xi_indices = []
        bunch_source_metadata = []

        # Calculate bunch sources and create adaptive grids if needed.
        s_d = ge.plasma_skin_depth(self.n_p * 1e-6)
        deposit_outliers_on_base_grid = False
        if self.use_adaptive_grids:
            store_plasma_history = True
            # Get radial grid resolution.
            if isinstance(self.adaptive_grid_nr, list):
                assert len(self.adaptive_grid_nr) == len(bunches), (
                    "Several resolutions for the adaptive grids have been "
                    "given, but they do not match the number of tracked "
                    "bunches"
                )
                nr_grids = self.adaptive_grid_nr
            else:
                nr_grids = [self.adaptive_grid_nr] * len(bunches)
            # Get radial extent.
            if isinstance(self.adaptive_grid_r_max, list):
                assert len(self.adaptive_grid_r_max) == len(bunches), (
                    "Several `r_max` for the adaptive grids have been "
                    "given, but they do not match the number of tracked "
                    "bunches"
                )
                r_max_grids = self.adaptive_grid_r_max
            else:
                r_max_grids = [self.adaptive_grid_r_max] * len(bunches)
            # Get radial extent limit.
            if isinstance(self.adaptive_grid_r_lim, list):
                assert len(self.adaptive_grid_r_lim) == len(bunches), (
                    "Several `r_lim` for the adaptive grids have been "
                    "given, but they do not match the number of tracked "
                    "bunches"
                )
                r_lim_grids = self.adaptive_grid_r_lim
            else:
                r_lim_grids = [self.adaptive_grid_r_lim] * len(bunches)
            # Check that the given extents are not larger than the limits.
            for r_lim, r_max in zip(r_lim_grids, r_max_grids):
                if r_lim is not None and r_max is not None:
                    if r_max > r_lim:
                        raise ValueError("`r_max` cannot be larger than `r_lim`")
            # Create adaptive grids for each bunch.
            bunches_with_grid: List[ParticleBunch] = []
            bunches_without_grid: List[ParticleBunch] = []
            for i, bunch in enumerate(bunches):
                if nr_grids[i] is not None:
                    bunches_with_grid.append(bunch)
                    if bunch.name not in self.bunch_grids:
                        self.bunch_grids[bunch.name] = AdaptiveGrid(
                            bunch.x,
                            bunch.y,
                            bunch.xi,
                            bunch.name,
                            nr_grids[i],
                            self.n_xi,
                            self.xi_fld,
                            r_max_grids[i],
                            r_lim_grids[i],
                        )
                else:
                    bunches_without_grid.append(bunch)
            # Calculate bunch sources at each grid.
            for bunch in bunches_with_grid:
                grid = self.bunch_grids[bunch.name]
                all_deposited = grid.calculate_bunch_source(
                    bunch, self.n_p, self.p_shape
                )
                bunch_source_arrays.append(grid.b_t_bunch)
                bunch_source_xi_indices.append(grid.i_grid)
                bunch_source_metadata.append(
                    np.array([grid.r_min_cell, grid.r_max_cell_guard, grid.dr]) / s_d
                )
                if not all_deposited:
                    self._reset_bunch_arrays()
                    deposit_bunch_charge(
                        bunch.x,
                        bunch.y,
                        bunch.xi,
                        bunch.q,
                        self.n_p,
                        self.n_r,
                        self.n_xi,
                        self.r_fld,
                        self.xi_fld,
                        self.dr,
                        self.dxi,
                        self.p_shape,
                        self.q_bunch,
                        r_min_deposit=grid.r_max,
                    )
                    deposit_outliers_on_base_grid = True

        else:
            bunches_without_grid = bunches
        # If not using adaptive grids, add all sources to the same array.
        if bunches_without_grid or deposit_outliers_on_base_grid:
            self._reset_bunch_arrays()
            for bunch in bunches_without_grid:
                deposit_bunch_charge(
                    bunch.x,
                    bunch.y,
                    bunch.xi,
                    bunch.q,
                    self.n_p,
                    self.n_r,
                    self.n_xi,
                    self.r_fld,
                    self.xi_fld,
                    self.dr,
                    self.dxi,
                    self.p_shape,
                    self.q_bunch,
                )
                

                print(bunches[0].w)






                q_new, _ = inverse_deposit_3d_distribution(
                    bunches[0].xi, bunches[0].x, bunches[0].y,
                    self.xi_fld[0], self.r_fld[0],
                    self.n_xi, self.n_r, self.dxi, self.dr,
                    self.q_bunch,
                    p_shape=self.p_shape,
                    use_ruyten=True,
                )
                
                
                # 0) Make a grid that counts macroparticles (smoothed)
                count_grid = np.zeros_like(self.q_bunch)
                
                ones = np.ones_like(bunches[0].q)  # unit weight per macroparticle
                
                # Deposit ones (NO k here; use deposit_3d_distribution directly)
                deposit_3d_distribution(
                    bunches[0].xi, bunches[0].x, bunches[0].y,
                    ones,
                    self.xi_fld[0],
                    self.r_fld[0],
                    self.n_xi,
                    self.n_r,
                    self.dxi,
                    self.dr,
                    count_grid,
                    p_shape=self.p_shape,
                    use_ruyten=True,
                )

                count_p, _ = inverse_deposit_3d_distribution(
                    bunches[0].xi, bunches[0].x, bunches[0].y,
                    self.xi_fld[0], self.r_fld[0],
                    self.n_xi, self.n_r, self.dxi, self.dr,
                    count_grid,
                    p_shape=self.p_shape,
                    use_ruyten=True,
                )



                k = 1.0 / (2 * np.pi * ct.e * self.dr * self.dxi * s_d * self.n_p)
                
                # convert back to charge-like quantity
                q_est = q_new / k

                eps = 1e-30
                w_est = np.abs(q_est / bunches[0].q_species) / np.maximum(count_p, eps)

                print(np.abs(q_est / bunches[0].q_species))
                print(count_p)

                bunches[0].w =  w_est
                print(bunches[0].w)


#            # ---- beam-loading only on first solve ----
#            if self._apply_beamloading_this_solve:
#                print(self.xi_fld)
#                print(self.dxi)
#                print([np.max(self.q_bunch),np.min(self.q_bunch)])
#    
#    
#                # ---- INSERT beam-loading shaper HERE ----
#                # Example: increase witness charge by 5% between xi=-200um and -100um
#                self._apply_beamloading_once_on_deposited_qbunch()
#                #self._apply_beamloading_shaper_on_qbunch(xi_min=-64e-6, xi_max=-64e-6+1*self.dxi,
#                #                                          scale=1, smooth=0)
#                print([np.max(self.q_bunch),np.min(self.q_bunch)])


            print(bunches[0].w)
            if not self._initial_condition_done:
                # ---- INITIAL CONDITION ONLY ----
                self._beamloading_initial_condition(
                    laser_a2,
                    radial_density,
                    store_plasma_history,
                    bunch_source_arrays,
                    bunch_source_xi_indices,
                    bunch_source_metadata,
                    bunches
                    )


            print(bunches[0].w)
            


#            if self._initial_condition_done:
#                calculate_bunch_source(self.q_bunch, self.n_r, self.n_xi, self.b_t_bunch)
#                bunch_source_arrays.append(self.b_t_bunch)
#                bunch_source_xi_indices.append(np.arange(self.n_xi))
#                bunch_source_metadata.append(
#                    np.array(
#                        [
#                            self.r_fld[0],
#                            self.r_fld[-1] + 2 * self.dr,  # r of last guard cell.
#                            self.dr,
#                        ]
#                    )
#                    / s_d
#                )


            if self._initial_condition_done:
                calculate_bunch_source(self.q_bunch, self.n_r, self.n_xi, self.b_t_bunch)
                if len(bunch_source_arrays) == 0:
                    bunch_source_arrays.append(self.b_t_bunch)
                    bunch_source_xi_indices.append(np.arange(self.n_xi))
                    bunch_source_metadata.append(
                        np.array([self.r_fld[0], self.r_fld[-1] + 2 * self.dr, self.dr]) / s_d
                    )
                else:
                    # overwrite base-grid source (don’t append duplicates)
                    bunch_source_arrays[0] = self.b_t_bunch






        if self._initial_condition_done:

            # Calculate rho only if requested in the diagnostics.
            calculate_rho = any("rho" in diag for diag in self.field_diags)
    
            # Calculate plasma wakefields
            self.pp = calculate_wakefields(
                laser_a2,
                self.r_max,
                self.xi_min,
                self.xi_max,
                self.n_r,
                self.n_xi,
                self.ppc,
                self.n_p,
                r_max_plasma=self.r_max_plasma,
                radial_density=radial_density,
                p_shape=self.p_shape,
                max_gamma=self.max_gamma,
                plasma_pusher=self.plasma_pusher,
                ion_motion=self.ion_motion,
                ion_mass=self.ion_mass,
                free_electrons_per_ion=self.free_electrons_per_ion,
                fld_arrays=self.fld_arrays,
                bunch_source_arrays=bunch_source_arrays,
                bunch_source_xi_indices=bunch_source_xi_indices,
                bunch_source_metadata=bunch_source_metadata,
                store_plasma_history=store_plasma_history,
                calculate_rho=calculate_rho,
                particle_diags=self.particle_diags,
            )
    
    
    
    #        if self._apply_beamloading_this_solve:
    #            E_z = self.fld_arrays[5]
    #            E_z_phys = E_z[2:-2, 2:-2]   # interior (no guard cells)
    #            print(E_z_phys[:,0])
    
    
    
    
            # Add bunch density to total density.
            if calculate_rho:
                rho_bunch = -self.q_bunch[2:-2, 2:-2] / (self.r_fld / s_d)
                self.rho[2:-2, 2:-2] += rho_bunch
    
            # Calculate fields on adaptive grids.
            if self.use_adaptive_grids:
                for _, grid in self.bunch_grids.items():
                    grid.calculate_fields(self.n_p, self.pp)
    
    #        # After first wake solve, disable one-time beam-loading effect
    #        if self._apply_beamloading_this_solve:
    #            self._apply_beamloading_this_solve = False
    
    
        # After first wake solve, disable one-time beam-loading effect
        self._initial_condition_done = True


    def _reset_bunch_arrays(self):
        """Reset to zero the bunch arrays of the base grid."""
        if self.t > self._t_reset_bunch_arrays:
            self.b_t_bunch[:] = 0.0
            self.q_bunch[:] = 0.0
            self._t_reset_bunch_arrays = self.t

    def _get_radial_density(self, z_current):
        """Get radial density profile function"""

        def radial_density(r):
            n_p = self.density_function(z_current, r)
            if self.parabolic_coefficient is not None:
                pc = self.parabolic_coefficient(z_current)
                n_p = n_p + n_p * pc * r**2
            return n_p

        return radial_density

    def _get_parabolic_coefficient_fn(self, parabolic_coefficient):
        """Get parabolic_coefficient profile function"""
        if isinstance(parabolic_coefficient, float):

            def uniform_parabolic_coefficient(z):
                return np.ones_like(z) * parabolic_coefficient

            return uniform_parabolic_coefficient
        elif callable(parabolic_coefficient):
            return parabolic_coefficient
        else:
            raise ValueError(
                "Type {} not supported for parabolic_coefficient.".format(
                    type(parabolic_coefficient)
                )
            )

    def _gather(self, x, y, z, t, ex, ey, ez, bx, by, bz, bunch_name):
        # If using adaptive grids, gather fields from them.
        if bunch_name in self.bunch_grids:
            grid = self.bunch_grids[bunch_name]
            grid.update_if_needed(x, y, z, self.n_p, self.pp)
            all_gathered = grid.gather_fields(x, y, z, ex, ey, ez, bx, by, bz)
            # If not all particles managed to gather from adaptive grid (for
            # example, because they escaped from it), try to gather from the
            # base grid.
            if not all_gathered:
                gather_main_fields_cyl_linear(
                    self.e_r,
                    self.e_z,
                    self.b_t,
                    self.xi_fld[0],
                    self.xi_fld[-1],
                    self.r_fld[0],
                    self.r_fld[-1],
                    self.dxi,
                    self.dr,
                    x,
                    y,
                    z,
                    ex,
                    ey,
                    ez,
                    bx,
                    by,
                    bz,
                    r_min_gather=grid.r_max_cell,
                )

        # Otherwise, use base implementation.
        else:
            super()._gather(x, y, z, t, ex, ey, ez, bx, by, bz, bunch_name)

    def _get_openpmd_diagnostics_data(self, global_time):
        diag_data = super()._get_openpmd_diagnostics_data(global_time)
        # Add fields from adaptive grids to openpmd diagnostics.
        if self.use_adaptive_grids:
            for _, grid in self.bunch_grids.items():
                grid_data = grid.get_openpmd_data(global_time, self.adaptive_grid_diags)
                diag_data["fields"] += grid_data["fields"]
                for field in grid_data["fields"]:
                    diag_data[field] = grid_data[field]
        # Add plasma particles to openpmd diagnostics.
        particle_diags = self._get_plasma_particle_diagnostics(global_time)
        diag_data = {**diag_data, **particle_diags}
        diag_data["species"] = list(particle_diags.keys())
        return diag_data

    def _get_plasma_particle_diagnostics(self, global_time):
        """Return dict with plasma particle diagnostics."""
        diag_dict = {}
        elec_name = "plasma_electrons"
        ions_name = "plasma_ions"
        if len(self.particle_diags) > 0:
            for name, hist in zip((elec_name, ions_name), self.pp):
                s_d = ge.plasma_skin_depth(self.n_p * 1e-6)

                if name == elec_name:
                    diag_dict[name] = {
                        "q": -ct.e,
                        "m": ct.m_e,
                        "geometry": "rz",
                    }
                else:
                    diag_dict[name] = {
                        "q": ct.e * self.free_electrons_per_ion,
                        "m": self.ion_mass,
                        "geometry": "rz",
                    }
                diag_dict[name]["name"] = name
                if "r" in self.particle_diags:
                    diag_dict[name]["r"] = hist["r_hist"] * s_d
                if "z" in self.particle_diags:
                    diag_dict[name]["z"] = hist["xi_hist"] * s_d + self.xi_max
                    diag_dict[name]["z_off"] = global_time * ct.c
                if "pr" in self.particle_diags:
                    diag_dict[name]["pr"] = (
                        hist["pr_hist"] * diag_dict[name]["mass"] * ct.c
                    )
                if "pz" in self.particle_diags:
                    diag_dict[name]["pz"] = (
                        hist["pz_hist"] * diag_dict[name]["mass"] * ct.c
                    )
                if "w" in self.particle_diags:
                    diag_dict[name]["w"] = hist["w_hist"] * (
                        self.n_p
                        if name == elec_name
                        else self.n_p / self.free_electrons_per_ion
                    )
                if "r_to_x" in self.particle_diags:
                    diag_dict[name]["r_to_x"] = hist["r_to_x_hist"]
                if "id" in self.particle_diags:
                    diag_dict[name]["id"] = hist["id_hist"]

        return diag_dict
