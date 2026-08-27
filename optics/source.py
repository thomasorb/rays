from __future__ import annotations

import numpy as np

from tracing.ray import Ray


class Source:

    def __init__(
        self,
        position=(0, 0, 0),
        direction=(1, 0, 0),
        wavelength=632.8e-9,
    ):
        self.position = np.asarray(
            position,
            dtype=float,
        )

        self.direction = np.asarray(
            direction,
            dtype=float,
        )

        self.direction /= np.linalg.norm(
            self.direction
        )

        self.wavelength = wavelength

    # ------------------------------------------------

    def emit(self):

        return Ray(
            self.position,
            self.direction,
            wavelength=self.wavelength,
        )
