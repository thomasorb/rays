from __future__ import annotations

import numpy as np

from .planar import PlanarElement


class Mirror(
    PlanarElement
):

    def interact(
        self,
        ray,
    ):

        n = self.normal

        n /= np.linalg.norm(n)

        d = ray.direction

        ray.direction = (
            d
            - 2.0 * np.dot(d, n) * n
        )

        ray.direction /= np.linalg.norm(
            ray.direction
        )

        return [ray]
