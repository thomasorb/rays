from __future__ import annotations

import numpy as np

from .planar import PlanarElement


class BeamSplitter(
    PlanarElement
):
    """
    Infinitely thin beam splitter.

    Reflection occurs exactly at
    the position specified by the user.

    Any glass behind the coating
    must be modelled separately
    using Window().
    """

    def __init__(
        self,
        R=0.5,
        T=0.5,
        max_generation_depth=2,
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

    # --------------------------------------------------

    def interact(
        self,
        ray,
    ):

        if (
            ray.generation
            >= self.max_generation_depth
        ):
            return [ray]

        normal = self.normal

        normal /= np.linalg.norm(
            normal
        )

        #
        # transmitted
        #

        transmitted = (
            ray.clone()
        )

        transmitted.branch_id = (
            ray.branch_id + ".T"
        )

        transmitted.generation += 1

        transmitted.complex_amplitude *= (
            np.sqrt(self.T)
        )

        #
        # reflected
        #

        reflected = (
            ray.clone()
        )

        reflected.branch_id = (
            ray.branch_id + ".R"
        )

        reflected.generation += 1

        d = reflected.direction

        reflected.direction = (
            d
            - 2.0
            * np.dot(
                d,
                normal,
            )
            * normal
        )

        reflected.direction /= (
            np.linalg.norm(
                reflected.direction
            )
        )

        reflected.complex_amplitude *= (
            1j
            * np.sqrt(
                self.R
            )
        )

        ray.is_alive = False

        return [
            reflected,
            transmitted,
        ]
