from __future__ import annotations

import numpy as np

from .planar import PlanarElement


class Window(
    PlanarElement
):

    def __init__(
        self,
        thickness,
        refractive_index=1.5,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs
        )

        self.thickness = thickness

        self.n = (
            refractive_index
        )

    # ------------------------------------------------

    def refract(
        self,
        direction,
        normal,
        n1,
        n2,
    ):
        """
        Snell law.
        """

        cos_i = (
            -np.dot(
                normal,
                direction
            )
        )

        eta = n1 / n2

        k = (
            1.0
            - eta**2
            * (
                1.0
                - cos_i**2
            )
        )

        if k < 0:
            return None

        return (
            eta * direction
            + (
                eta * cos_i
                - np.sqrt(k)
            )
            * normal
        )

    # ------------------------------------------------

    def interact(
        self,
        ray,
    ):
        """
        Plane parallel plate.
        """

        n = (
            self.transform.normal
        )

        d0 = ray.direction

        d1 = self.refract(
            d0,
            n,
            1.0,
            self.n,
        )

        if d1 is None:
            ray.is_alive = False
            return

        #
        # propagate inside glass
        #

        ray.propagate(
            self.thickness,
            refractive_index=self.n,
        )

        d2 = self.refract(
            d1,
            -n,
            self.n,
            1.0,
        )

    
        ray.direction = d2
        
        ray.direction /= np.linalg.norm(
            ray.direction
        )
        
        return [ray]
