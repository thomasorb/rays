from __future__ import annotations

import numpy as np

from .wavefront_error import (
    WavefrontError
)


class ZernikeFigureError(
    WavefrontError
):
    """
    Random low-order Zernike figure error.

    The generated figure is scaled so that
    its PV matches the requested specification.

    Current modes:

        Z2 : Tilt X
        Z3 : Tilt Y
        Z4 : Defocus
        Z5 : Astigmatism 0°
        Z6 : Astigmatism 45°
    """

    def __init__(
        self,
        pupil_radius,
        target_pv_opd,
        seed=None,
    ):

        self.pupil_radius = (
            float(pupil_radius)
        )

        self.target_pv_opd = (
            float(target_pv_opd)
        )

        self.rng = np.random.default_rng(
            seed
        )

        self.coeffs = None

        self.randomize()

    # ==================================================
    # Factory constructors
    # ==================================================

    @classmethod
    def mirror_lambda_pv(
        cls,
        wavelength,
        ratio,
        pupil_radius,
        seed=None,
    ):
        """
        Mirror specified as λ/N PV.

        Reflection doubles OPD.
        """

        pv_surface = (
            wavelength / ratio
        )

        pv_opd = (
            2.0 * pv_surface
        )

        return cls(
            pupil_radius=pupil_radius,
            target_pv_opd=pv_opd,
            seed=seed,
        )

    # --------------------------------------------------

    @classmethod
    def beamsplitter_lambda_pv(
        cls,
        wavelength,
        ratio,
        pupil_radius,
        seed=None,
    ):
        """
        Beamsplitter specified as λ/N PV.

        First approximation:
        no factor 2.
        """

        pv_surface = (
            wavelength / ratio
        )

        pv_opd = pv_surface

        return cls(
            pupil_radius=pupil_radius,
            target_pv_opd=pv_opd,
            seed=seed,
        )

    # ==================================================
    # Internal helpers
    # ==================================================

    def _evaluate_zernikes(
        self,
        x,
        y,
        coeffs,
    ):

        r = (
            np.sqrt(
                x*x + y*y
            )
            / self.pupil_radius
        )

        theta = np.arctan2(
            y,
            x,
        )

        Z2 = (
            r * np.cos(theta)
        )

        Z3 = (
            r * np.sin(theta)
        )

        Z4 = (
            2.0*r*r - 1.0
        )

        Z5 = (
            r*r
            * np.cos(
                2.0*theta
            )
        )

        Z6 = (
            r*r
            * np.sin(
                2.0*theta
            )
        )

        c2, c3, c4, c5, c6 = coeffs

        return (
            c2*Z2
            + c3*Z3
            + c4*Z4
            + c5*Z5
            + c6*Z6
        )

    # --------------------------------------------------

    def _measure_pv(
        self,
        coeffs,
        npts=256,
    ):

        rmax = self.pupil_radius

        x = np.linspace(
            -rmax,
             rmax,
             npts,
        )

        y = np.linspace(
            -rmax,
             rmax,
             npts,
        )

        X, Y = np.meshgrid(
            x,
            y,
        )

        R = np.sqrt(
            X**2 + Y**2
        )

        mask = (
            R <= rmax
        )

        values = (
            self._evaluate_zernikes(
                X[mask],
                Y[mask],
                coeffs,
            )
        )

        return (
            np.max(values)
            - np.min(values)
        )

    # ==================================================
    # WavefrontError API
    # ==================================================

    def randomize(
        self,
    ):

        coeffs = self.rng.normal(
            size=5
        )

        pv_current = (
            self._measure_pv(
                coeffs
            )
        )

        if pv_current > 0:

            scale = (
                self.target_pv_opd
                / pv_current
            )

            coeffs *= scale

        self.coeffs = coeffs

    # --------------------------------------------------

    def opd(
        self,
        x,
        y,
    ):

        r = (
            np.sqrt(
                x*x + y*y
            )
            / self.pupil_radius
        )

        if r > 1.0:
            return 0.0

        return (
            self._evaluate_zernikes(
                x,
                y,
                self.coeffs,
            )
        )

    # ==================================================
    # Diagnostics
    # ==================================================

    def statistics(
        self,
        npts=256,
    ):

        rmax = self.pupil_radius

        x = np.linspace(
            -rmax,
             rmax,
             npts,
        )

        y = np.linspace(
            -rmax,
             rmax,
             npts,
        )

        X, Y = np.meshgrid(
            x,
            y,
        )

        R = np.sqrt(
            X**2 + Y**2
        )

        mask = (
            R <= rmax
        )

        values = (
            self._evaluate_zernikes(
                X[mask],
                Y[mask],
                self.coeffs,
            )
        )

        rms = np.std(values)

        pv = (
            np.max(values)
            - np.min(values)
        )

        return {
            "rms_nm": rms * 1e9,
            "pv_nm": pv * 1e9,
        }

    # --------------------------------------------------

    def plot(
        self,
        npts=256,
        cmap="RdBu_r",
    ):

        import matplotlib.pyplot as plt

        rmax = self.pupil_radius

        x = np.linspace(
            -rmax,
             rmax,
             npts,
        )

        y = np.linspace(
            -rmax,
             rmax,
             npts,
        )

        X, Y = np.meshgrid(
            x,
            y,
        )

        R = np.sqrt(
            X**2 + Y**2
        )

        mask = (
            R <= rmax
        )

        OPD = np.full(
            X.shape,
            np.nan,
        )

        OPD[mask] = (
            self._evaluate_zernikes(
                X[mask],
                Y[mask],
                self.coeffs,
            )
        )

        stats = self.statistics()

        plt.figure(
            figsize=(6,5)
        )

        im = plt.imshow(
            OPD * 1e9,
            origin="lower",
            extent=[
                -rmax,
                 rmax,
                -rmax,
                 rmax,
            ],
            cmap=cmap,
        )

        plt.colorbar(
            im,
            label="OPD [nm]",
        )

        plt.title(
            f"PV={stats['pv_nm']:.1f} nm   "
            f"RMS={stats['rms_nm']:.1f} nm"
        )

        plt.xlabel("x")
        plt.ylabel("y")

        plt.axis("equal")

        plt.tight_layout()

        plt.show()
