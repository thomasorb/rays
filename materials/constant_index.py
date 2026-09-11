from __future__ import annotations

from .material import Material


class ConstantIndex(
    Material
):

    def __init__(
        self,
        n,
        name=None,
    ):

        self.n0 = float(n)

        if name is None:

            name = f"n={n}"

        self.name = name

    def n(
        self,
        wavelength,
    ):
        return self.n0
