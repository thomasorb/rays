from __future__ import annotations

from .material import Material


class SellmeierMaterial(
    Material
):

    def __init__(
        self,
        B1,
        B2,
        B3,
        C1,
        C2,
        C3,
        name="Sellmeier",
    ):

        self.B1 = B1
        self.B2 = B2
        self.B3 = B3

        self.C1 = C1
        self.C2 = C2
        self.C3 = C3

        self.name = name

    def n(
        self,
        wavelength,
    ):
        """
        wavelength in meters
        """

        lam_um = (
            wavelength * 1e6
        )

        lam2 = lam_um**2

        n2 = (
            1
            +
            self.B1 * lam2 /
            (lam2 - self.C1)
            +
            self.B2 * lam2 /
            (lam2 - self.C2)
            +
            self.B3 * lam2 /
            (lam2 - self.C3)
        )

        return n2**0.5
