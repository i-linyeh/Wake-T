import shutil
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
from openpmd_viewer import OpenPMDTimeSeries

from wake_t import ParticleBunch, PlasmaStage, Beamline
from wake_t.particles.deposition import deposit_3d_distribution


def test_salame(show=False, plot=False):
    """
    Test that SALAME reduces the witness weighted <Ez> residual to an
    acceptable level for a two-bunch Wake-T setup.

    The test performs two runs:

        (a) a reference run without SALAME to define Ez_target from the
            last nonzero weighted <Ez> value of the witness bunch.

        (b) a SALAME-enabled run, from which the weighted <Ez> profile
            is compared against Ez_target.

    The test passes if

        residual_sum < 0.2

    where residual_sum is the average relative absolute deviation of the
    nonzero weighted <Ez> samples from Ez_target.
    """
    np.random.seed(0)

    diag_dir = "diags_salame_multi_bunch_test"

    try:
        ez_target = run_case_and_get_target(
            do_salame=False,
            diag_dir=diag_dir,
        )

        z, i_z, ez_bunch_weighted, z_part = run_case_and_get_weighted_ez(
            do_salame=True,
            diag_dir=diag_dir,
        )

        ez_bunch_weighted_nonzero = ez_bunch_weighted[ez_bunch_weighted != 0]

        residual_sum = np.sum(np.abs(ez_bunch_weighted_nonzero - ez_target)) / (
            np.abs(ez_target) * len(ez_bunch_weighted_nonzero)
        )

        if residual_sum >= 0.2:
            raise AssertionError(f"Residual too large: {residual_sum}")

        if show:
            print(f"Ez_target = {ez_target:.6e}")
            print(f"residual_sum = {residual_sum:.6e}")

        if plot:
            plot_current_and_weighted_ez(z, i_z, ez_bunch_weighted, z_part)

    finally:
        shutil.rmtree(diag_dir, ignore_errors=True)


def run_case_and_get_target(do_salame, diag_dir):
    """
    Run one simulation case and return Ez_target, defined as the last nonzero
    value of the witness weighted <Ez> profile.
    """
    z, i_z, ez_bunch_weighted, z_part = run_case_and_get_weighted_ez(
        do_salame=do_salame,
        diag_dir=diag_dir,
    )
    ez_bunch_weighted_nonzero = ez_bunch_weighted[ez_bunch_weighted != 0]
    return ez_bunch_weighted_nonzero[-1]


def run_case_and_get_weighted_ez(do_salame, diag_dir):
    """
    Run one simulation case and return:
        z, witness current I_z(z), weighted <Ez>(z), and witness particle z.
    """
    shutil.rmtree(diag_dir, ignore_errors=True)

    plasma_plateau, bunch_dri, bunch_wit = build_test_setup()

    bunch_wit.do_salame = do_salame
    bunch_wit.salame_n_iter = 100
    bunch_wit.salame_relative_tolerance = 1e-6
    bunch_wit.use_avg_psi = True

    beamline = Beamline([plasma_plateau])
    beamline.track(
        [bunch_dri, bunch_wit],
        opmd_diag=True,
        show_progress_bar=False,
        diag_dir=diag_dir,
    )

    return get_witness_current_and_weighted_ez(diag_dir)


def build_test_setup():
    """
    Build the plasma stage, driver bunch, and witness bunch used in the test.
    """
    n_p0 = 1.7e23
    kp = np.sqrt(n_p0 * sc.e**2 / (sc.epsilon_0 * sc.m_e * sc.c**2))

    l_plateau = 1e-3
    l_ramp = 0
    l_total = l_plateau + 2 * l_ramp

    def density_profile(z, r):
        n = n_p0 * np.ones_like(z)
        z0 = l_ramp + l_plateau
        n = np.where(z <= -155e-6, 1e-10 * n_p0, n)
        n = np.where(z >= -155e-6 + z0 + l_ramp, 1e-10 * n_p0, n)
        return n

    bunch_dri = build_driver_bunch(kp)
    bunch_wit, sx0, sy0 = build_witness_bunch(kp)

    r_max = 144e-6
    r_max_plasma = 108e-6
    l_box = 200e-6
    xi_max = 45e-6
    xi_min = xi_max - l_box
    dz = 1 / kp / 80
    nz = int(l_box / dz)
    dr = 1 / kp / 40
    nr = int(r_max / dr)
    dz_fields = 200e-6

    ppc = [[r_max_plasma, 2]]

    plasma_plateau = PlasmaStage(
        length=l_total,
        density=density_profile,
        wakefield_model="quasistatic_2d",
        ion_motion=False,
        n_out=50,
        laser=None,
        laser_evolution=True,
        r_max=r_max,
        r_max_plasma=r_max_plasma,
        xi_min=xi_min,
        xi_max=xi_max,
        n_r=nr,
        n_xi=nz,
        dz_fields=dz_fields,
        ppc=ppc,
        laser_envelope_substeps=4,
        laser_envelope_nxi=nz * 4,
        max_gamma=10,
        field_diags=["rho", "E", "B", "a", "q_bunch", "q_bunch_wit"],
        use_adaptive_grids=False,
    )

    return plasma_plateau, bunch_dri, bunch_wit


def build_driver_bunch(kp):
    q_tot = 1000e-12
    ene_sp = 0.1
    n_emitt_x = 0.1e-6
    n_emitt_y = 0.1e-6
    kin_energy_GeV = 100.0
    l_beam = 5.453219e-6
    n_part = int(1e6)

    mc2_GeV = sc.m_e * sc.c**2 / sc.electron_volt / 1e9
    gamma_beam = 1 + kin_energy_GeV / mc2_GeV

    betax0 = np.sqrt(2 * gamma_beam) / kp
    sx0 = np.sqrt(n_emitt_x * betax0 / gamma_beam)
    sy0 = np.sqrt(n_emitt_y * betax0 / gamma_beam)

    zmin = -l_beam / 2
    zmax = l_beam / 2

    z = np.random.uniform(zmin, zmax, n_part)
    x = sx0 * np.random.standard_normal(n_part)
    y = sy0 * np.random.standard_normal(n_part)

    s_g = ene_sp * gamma_beam / 100
    gamma = np.random.normal(gamma_beam, s_g, n_part)

    s_ux = n_emitt_x / sx0
    s_uy = n_emitt_y / sy0
    ux = s_ux * np.random.standard_normal(n_part)
    uy = s_uy * np.random.standard_normal(n_part)

    uz = np.sqrt((gamma**2 - 1) - ux**2 - uy**2)

    q = np.ones(n_part) * q_tot / n_part
    w = np.abs(q / sc.e)

    return ParticleBunch(w, x, y, z, ux, uy, uz, name="bunch_dri")


def build_witness_bunch(kp):
    q_tot = 100e-12
    ene_sp = 0.1
    n_emitt_x = 0.1e-6
    n_emitt_y = 0.1e-6
    kin_energy_GeV = 100.0
    l_beam = 5.453219e-6
    d_beam = 62.973740e-6 * 1.2
    n_part = int(1e6)

    mc2_GeV = sc.m_e * sc.c**2 / sc.electron_volt / 1e9
    gamma_beam = 1 + kin_energy_GeV / mc2_GeV

    betax0 = np.sqrt(2 * gamma_beam) / kp
    sx0 = np.sqrt(n_emitt_x * betax0 / gamma_beam)
    sy0 = np.sqrt(n_emitt_y * betax0 / gamma_beam)

    zc = -d_beam - l_beam / 3
    zmin = zc - l_beam / 2
    zmax = zc + l_beam / 2

    z = np.random.uniform(zmin, zmax, n_part)
    x = sx0 * np.random.standard_normal(n_part)
    y = sy0 * np.random.standard_normal(n_part)

    s_g = ene_sp * gamma_beam / 100
    gamma = np.random.normal(gamma_beam, s_g, n_part)

    s_ux = n_emitt_x / sx0
    s_uy = n_emitt_y / sy0
    ux = s_ux * np.random.standard_normal(n_part)
    uy = s_uy * np.random.standard_normal(n_part)

    uz = np.sqrt((gamma**2 - 1) - ux**2 - uy**2)

    q = np.ones(n_part) * q_tot / n_part
    w = np.abs(q / sc.e)

    bunch_wit = ParticleBunch(w, x, y, z, ux, uy, uz, name="bunch_wit")
    return bunch_wit, sx0, sy0


def get_witness_current_and_weighted_ez(diag_dir, iteration=0):
    ts = OpenPMDTimeSeries(f"./{diag_dir}/hdf5/")

    ez, info_ez = ts.get_field(iteration=iteration, field="E", coord="z")
    z = info_ez.z
    r = info_ez.r

    dz = z[1] - z[0]
    dr = r[1] - r[0]
    nz = ez.shape[1]
    nr = ez.shape[0]

    x, y, z_part, uz = ts.get_particle(
        var_list=["x", "y", "z", "uz"],
        iteration=iteration,
        species="bunch_wit",
    )
    q, weight, m = ts.get_particle(
        var_list=["charge", "w", "mass"],
        iteration=iteration,
        species="bunch_wit",
    )

    rho = np.zeros((nz + 4, int(nr / 2) + 4))
    deposit_3d_distribution(
        z_part,
        x,
        y,
        q * weight,
        z[0],
        r[int(nr / 2)],
        nz,
        int(nr / 2),
        dz,
        dr,
        rho,
        p_shape="z0r1",
    )

    rho_phys = rho[2:-2, 2:-2]
    g_xi = np.sum(rho_phys, axis=1)
    i_z = g_xi / dz * sc.c

    ez_zr = ez[int(nr / 2) :, :].T
    qb = rho_phys
    w = np.abs(qb)
    wsum = np.sum(w, axis=1)

    ez_bunch_weighted = np.zeros(ez_zr.shape[0], dtype=float)
    mask = wsum > 0
    ez_bunch_weighted[mask] = np.sum(ez_zr[mask, :] * w[mask, :], axis=1) / wsum[mask]

    return z, i_z, ez_bunch_weighted, z_part


def plot_current_and_weighted_ez(z, i_z, ez_bunch_weighted, z_part):
    fig, ax1 = plt.subplots(figsize=(6, 3), constrained_layout=False)

    ax1.plot(
        z * 1e6,
        i_z,
        "r",
        marker="o",
        markersize=3,
        linewidth=1,
        label="|I| [A]",
    )
    ax1.tick_params(axis="y", colors="red")
    ax1.set_xlabel(r"$\xi$ [$\mu$m]")
    ax1.set_ylabel(r"|I| [A]", color="red")
    ax1.set_xlim(
        (z_part.min() - abs(z_part.min()) * 0.05) / 1e-6,
        (z_part.max() + abs(z_part.max()) * 0.05) / 1e-6,
    )
    ax1.set_ylim(-22000, 1200)

    ax2 = ax1.twinx()
    ax2.plot(
        z * 1e6,
        ez_bunch_weighted,
        "blue",
        marker="o",
        markersize=3,
        linewidth=1,
        linestyle="--",
        label=r"Weighted $\langle E_z\rangle$ [V/m]",
    )
    ax2.set_ylabel(r"Weighted $\langle E_z\rangle$ [V/m]", color="blue")
    ax2.tick_params(axis="y", colors="blue")
    ax2.set_ylim(-5e10, 3e9)

    plt.show()


if __name__ == "__main__":
    test_salame(show=False, plot=False)
