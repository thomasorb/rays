from __future__ import annotations

from materials.air import AIR

from .circular_optical_interface import (
    CircularOpticalInterface
)


class CircularBeamSplitter(
    CircularOpticalInterface
):
    """
    Infinitely thin circular beam splitter.
    """

    def __init__(
        self,

        R=0.5,
        T=0.5,

        *args,
        **kwargs,
    ):

        super().__init__(

            material1=AIR,
            material2=AIR,

            R=R,
            T=T,

            *args,
            **kwargs,
        )
