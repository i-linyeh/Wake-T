import os
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt

from openpmd_viewer import OpenPMDTimeSeries


tests_output_folder = "./tests_output"


def test_salame_current_profile_flattening(plot=False):
    """
    This test checks that the SALAME-generated charge profile produces
    a bunch-weighted longitudinal electric field that is sufficiently flat
    across the nonzero part of the bunch.

    The residual is defined as the average relative deviation of the
    bunch-weighted Ez from its value at the bunch tail.
    """
    output_folder = os.path.join(tests_output_folder, "salame_output")
    diag_dir = os.path.join(output_folder, "hdf5")

    # Load openPMD diagnostics.
    ts = OpenPMDTimeSeries(diag_dir)

    # Select iteration.
    it = 0

    # Read longitudinal electric field.
    Ez, info_Ez = ts.get_field(iteration=it, field="E", coord="z")
    z = info_Ez.z
    r = info_Ez.r

    dz = z[1] - z[0]
    nr = Ez.shape[0]

    # Read SALAME bunch charge profile.
    q_bunch, _ = ts.get_field(iteration=it, field="charge_profile_salame")

    # Convert to (z, r) ordering for easier processing.
    Ez_zr = Ez.T
    q_bunch_zr = q_bunch.T

    # Compute line charge and current profile.
    g_xi = np.sum(q_bunch_zr[:, int(nr / 2) :], axis=1)
    I_z = g_xi / dz * sc.c

    # Compute bunch-weighted <Ez>.
    w = np.abs(q_bunch_zr)
    wsum = np.sum(w, axis=1)

    Ez_bunch_weighted = np.zeros(Ez_zr.shape[0], dtype=float)
    mask = wsum > 0
    Ez_bunch_weighted[mask] = np.sum(Ez_zr[mask, :] * w[mask, :], axis=1) / wsum[mask]

    # Restrict to nonzero bunch region.
    indices = np.flatnonzero(wsum)
    Ez_target = Ez_bunch_weighted[indices[-1]]
    Ez_nonzero = Ez_bunch_weighted[indices]

    # Compute flattening residual.
    residual = np.sum(np.abs(Ez_nonzero - Ez_target)) / (
        np.abs(Ez_target) * len(Ez_nonzero)
    )

    # Check that residual is sufficiently small.
    assert residual < 0.2

    # Optional diagnostic plot.
    if plot:
        fig, ax1 = plt.subplots(figsize=(6, 3), constrained_layout=False)

        # Left axis: current profile.
        ax1.plot(
            z * 1e6,
            I_z,
            "r",
            marker="o",
            markersize=3,
            linewidth=1,
            label="Current [A]",
        )
        ax1.set_xlabel(r"$\xi$ [$\mu$m]")
        ax1.set_ylabel(r"Current [A]", color="red")
        ax1.tick_params(axis="y", colors="red")
        ax1.set_xlim(z[indices[0]] / 1e-6 - 2, z[indices[-1]] / 1e-6 + 2)
        ax1.set_ylim(-22000, 1200)

        # Right axis: bunch-weighted Ez.
        ax2 = ax1.twinx()
        ax2.plot(
            z * 1e6,
            Ez_bunch_weighted,
            "b",
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
    test_salame_current_profile_flattening(plot=True)
