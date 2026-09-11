from __future__ import annotations

from .sellmeier import (
    SellmeierMaterial
)


class FusedSilica(
    SellmeierMaterial
):

    def __init__(
        self,
    ):

        super().__init__(

            B1=0.6961663,
            B2=0.4079426,
            B3=0.8974794,

            C1=0.0684043**2,
            C2=0.1162414**2,
            C3=9.896161**2,

            name="Fused Silica",
        )
