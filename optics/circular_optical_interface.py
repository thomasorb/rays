from __future__ import annotations

from .circular_planar import (
    CircularPlanarElement
)

from .base_optical_interface import (
    BaseOpticalInterface
)


class CircularOpticalInterface(
    BaseOpticalInterface,
    CircularPlanarElement,
):
    pass
