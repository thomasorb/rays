from __future__ import annotations

from geometry.transform import Transform
from geometry.plane import Plane


class OpticalElement:
    """
    Base class for all optical elements.
    """

    def __init__(
        self,
        position=None,
        rotation=None,
        name=None,
    ):
        self.name = name

        self.transform = Transform(
            position=position,
            rotation=rotation,
        )

        self.surface = Plane(
            self.transform
        )

    # --------------------------------------------------

    def intersect(self, ray):
        return self.surface.intersect(
            ray.origin,
            ray.direction,
        )

    # --------------------------------------------------

    def interact(self, ray):
        raise NotImplementedError
