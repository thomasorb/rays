from __future__ import annotations

import numpy as np

from .planar import PlanarElement


class Detector(
    PlanarElement
):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs
        )

        self.hit = False

        self.hit_position = None

        self.last_phase = None
        self.last_opl = None

    # ------------------------------------------------

    def interact(
        self,
        ray,
    ):
        """
        Record impact.
        """

        local_point = (
            self.transform
            .world_to_local_point(
                ray.origin
            )
        )

        self.hit = True

        self.hit_position = local_point

        self.last_phase = ray.phase

        self.last_opl = (
            ray.optical_length
        )

        ray.is_alive = False
