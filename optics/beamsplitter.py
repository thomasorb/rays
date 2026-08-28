from __future__ import annotations

import numpy as np

from .planar import PlanarElement


class BeamSplitter(
    PlanarElement
):

    def __init__(
        self,
        R=0.5,
        T=0.5,
        max_generation_depth=100,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs
        )

        self.R = R
        self.T = T

        self.max_generation_depth = (
            max_generation_depth
        )
    # ----------------------------------------------------------

    def interact(
        self,
        ray,
    ):

        self.apply_wavefront_error(
            ray
        )

        if (
            ray.generation
            >= self.max_generation_depth
        ):
            return [ray]

        n = self.normal

        n /= np.linalg.norm(n)

        #
        # Transmission
        #

        transmitted = ray.clone()

        transmitted.branch_id = (
            ray.branch_id + ".T"
        )

        transmitted.generation += 1

        transmitted.complex_amplitude *= (
            np.sqrt(self.T)
        )

        #
        # Reflection
        #

        reflected = ray.clone()

        reflected.branch_id = (
            ray.branch_id + ".R"
        )

        reflected.generation += 1

        d = reflected.direction

        reflected.direction = (
            d
            - 2*np.dot(d,n)*n
        )

        reflected.direction /= (
            np.linalg.norm(
                reflected.direction
            )
        )

        reflected.complex_amplitude *= (
            1j*np.sqrt(self.R)
        )

        ray.is_alive = False

        return [
            reflected,
            transmitted,
        ]
