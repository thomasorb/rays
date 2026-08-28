from __future__ import annotations

import numpy as np

from .wavefront_error import (
    WavefrontError
)


class FigureError(
    WavefrontError
):
    """
    Random figure error model.

    Current model:

        constant OPD over entire optic

    but randomized between simulations.

    This is a good first approximation
    for modulation efficiency studies.
    """

    def __init__(
        self,
        rms_opd,
        seed=None,
    ):

        self.rms_opd = float(
            rms_opd
        )

        self.rng = np.random.default_rng(
            seed
        )

        self.current_opd = 0.0

        self.randomize()

    # -------------------------------------------------

    @classmethod
    def mirror_lambda_pv(
        cls,
        wavelength,
        ratio,
        seed=None,
    ):

        pv_surface = (
            wavelength / ratio
        )

        rms_surface = (
            pv_surface / 4.0
        )

        rms_opd = (
            2.0 * rms_surface
        )

        return cls(
            rms_opd=rms_opd,
            seed=seed,
        )

    @classmethod
    def beamsplitter_lambda_pv(
        cls,
        wavelength,
        ratio,
        seed=None,
    ):
        """
        First approximation.

        Unlike mirrors, do not double
        the wavefront error.
        """

        pv_surface = (
            wavelength / ratio
        )

        rms_surface = (
            pv_surface / 4.0
        )

        rms_opd = rms_surface

        return cls(
            rms_opd=rms_opd,
            seed=seed,
        )

    # -------------------------------------------------

    def randomize(
        self,
    ):

        self.current_opd = (
            self.rng.normal(
                0.0,
                self.rms_opd,
            )
        )

    # -------------------------------------------------

    def opd(
        self,
        x,
        y,
    ):
        return self.current_opd

    # -------------------------------------------------

    @property
    def rms_nm(
        self,
    ):
        return (
            self.rms_opd
            * 1e9
        )
