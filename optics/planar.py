from __future__ import annotations

import numpy as np

from .base import OpticalElement


class PlanarElement(
    OpticalElement
):
    """
    Generic planar optic.

    Local plane:

        z = 0
    """

    def __init__(
        self,
        width=50.0,
        height=50.0,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        self.width = width
        self.height = height

    # --------------------------------------------------

    def get_local_outline(self):

        w = self.width / 2
        h = self.height / 2

        return np.array([
            [-w, -h, 0],
            [ w, -h, 0],
            [ w,  h, 0],
            [-w,  h, 0],
            [-w, -h, 0],
        ])

    # --------------------------------------------------

    def get_outline(self):

        local = (
            self.get_local_outline()
        )

        return np.array([
            self.transform
                .local_to_world_point(
                    p
                )
            for p in local
        ])

    
    def contains_local_point(
        self,
        point,
    ):
        """
        Check if a local point lies inside
        the finite optic.
        """

        x = point[0]
        y = point[1]

        return (
            abs(x) <= self.width / 2
            and
            abs(y) <= self.height / 2
        )

    def intersect(
            self,
            ray,
        ):
        """
        Finite planar surface intersection.
        """

        distance = self.surface.intersect(
            ray.origin,
            ray.direction,
        )

        if distance is None:
            return None

        hit_point = (
            ray.origin
            + distance * ray.direction
        )

        local_hit = (
            self.transform
            .world_to_local_point(
                hit_point
            )
        )

        if not self.contains_local_point(
            local_hit
        ):
            return None

        return distance
