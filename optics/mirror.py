from __future__ import annotations

from materials.constant_index import (
    ConstantIndex
)

from .optical_interface import (
    OpticalInterface
)

from materials.air import AIR

class Mirror(
    OpticalInterface
):

    def __init__(
        self,
        *args,
        **kwargs,
    ):

       
        super().__init__(

            material1=AIR,
            material2=AIR,

            R=1.0,
            T=0.0,

            *args,
            **kwargs,
        )
