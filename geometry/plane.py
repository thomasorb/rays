from __future__ import annotations

import numpy as np

from .transform import Transform


class Plane:
    """
    Infinite plane.

    Local equation:

        z = 0
    """

    def __init__(
        self,
        transform: Transform,
    ):
        self.transform = transform

    def intersect(
        self,
        ray_origin,
        ray_direction,
    ):
        """
        Returns distance along ray.

        None if no hit.
        """

        origin_local = (
            self.transform.world_to_local_point(
                ray_origin
            )
        )

        direction_local = (
            self.transform.world_to_local_vector(
                ray_direction
            )
        )

        dz = direction_local[2]

        if abs(dz) < 1e-12:
            return None

        t = -origin_local[2] / dz

        EPSILON = 1e-9
        if t <= EPSILON:
            return None

        return t
