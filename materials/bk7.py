from __future__ import annotations

from .sellmeier import (
    SellmeierMaterial
)


class BK7(
    SellmeierMaterial
):

    def __init__(
        self,
    ):

        super().__init__(

            B1=1.03961212,
            B2=0.231792344,
            B3=1.01046945,

            C1=0.00600069867,
            C2=0.0200179144,
            C3=103.560653,

            name="BK7",
        )
