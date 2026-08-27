from __future__ import annotations

import numpy as np

from .base import OpticalElement


class Aperture(
    OpticalElement
):

    def __init__(
        self,
        radius,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs
        )

        self.radius = radius

    # ------------------------------------------------

    def interact(
        self,
        ray,
    ):
        p = (
            self.transform
            .world_to_local_point(
                ray.origin
            )
        )

        r = np.sqrt(
            p[0]**2
            + p[1]**2
        )

        if r > self.radius:
            ray.is_alive = False
