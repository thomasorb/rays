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

        #
        # Stop beam proliferation
        #

        if (
            ray.generation
            >= self.max_generation_depth
        ):
            return [ray]

        n = self.normal
        n /= np.linalg.norm(n)

        #
        # transmitted branch
        #

        transmitted = ray.clone()

        transmitted.branch_id = (
            ray.branch_id + ".T"
        )

        transmitted.amplitude *= (
            np.sqrt(self.T)
        )

        #
        # reflected branch
        #

        reflected = ray.clone()

        reflected.branch_id = (
            ray.branch_id + ".R"
        )

        d = reflected.direction

        reflected.direction = (
            d
            - 2.0
            * np.dot(d, n)
            * n
        )

        reflected.direction /= (
            np.linalg.norm(
                reflected.direction
            )
        )

        reflected.amplitude *= (
            np.sqrt(self.R)
        )

        ray.is_alive = False

        return [
            reflected,
            transmitted,
        ]
