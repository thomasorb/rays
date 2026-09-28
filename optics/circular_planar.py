from __future__ import annotations

import numpy as np

from .base import OpticalElement


class CircularPlanarElement(
    OpticalElement
):
    """
    Circular planar optic.

    Local plane:

        z = 0
    """

    def __init__(
        self,
        diameter=25.0,
        *args,
        **kwargs,
    ):

        super().__init__(
            *args,
            **kwargs,
        )

        self.diameter = float(
            diameter
        )

    # --------------------------------------------------

    @property
    def radius(
        self,
    ):
        return (
            self.diameter / 2
        )

    # --------------------------------------------------

    def get_local_outline(
        self,
        n_points=128,
    ):
        """
        Circular outline in local coordinates.
        """

        theta = np.linspace(
            0,
            2*np.pi,
            n_points,
        )

        r = self.radius

        return np.column_stack([
            r * np.cos(theta),
            r * np.sin(theta),
            np.zeros_like(theta),
        ])

    # --------------------------------------------------

    def get_outline(
        self,
    ):
        """
        World-space outline.
        """

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

    # --------------------------------------------------

    def contains_local_point(
        self,
        point,
    ):
        """
        Finite circular aperture test.
        """

        x = point[0]
        y = point[1]

        return (
            x*x + y*y
            <= self.radius**2
        )

    # --------------------------------------------------

    def intersect(
        self,
        ray,
    ):
        """
        Plane intersection
        + circular aperture clipping.
        """

        distance = (
            self.surface.intersect(
                ray.origin,
                ray.direction,
            )
        )

        if distance is None:

            return None

        hit_point = (

            ray.origin

            + distance
            * ray.direction
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
