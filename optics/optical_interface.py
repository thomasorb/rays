from __future__ import annotations

from .planar import (
    PlanarElement
)

from .base_optical_interface import (
    BaseOpticalInterface
)


class OpticalInterface(
    BaseOpticalInterface,
    PlanarElement,
):
    pass
