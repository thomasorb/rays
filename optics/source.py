from __future__ import annotations

import numpy as np

from tracing.ray import Ray
from .base import Optic


class Source(
    Optic
):

    def __init__(
        self,
        position=(0, 0, 0),
        direction=(1, 0, 0),
        wavelength=632.8e-9,
    ):
        self.set_position(
            position
        )
        self.set_direction(
            direction
        )
        self.set_wavelength(
            wavelength
        )

    # ------------------------------------------------

    def set_position(
        self,
        position,
    ):
        self.position = np.asarray(
            position,
            dtype=float,
        )

    def set_direction(
        self,
        direction,
    ):
        direction = np.asarray(
            direction,
            dtype=float,
        )

        norm = np.linalg.norm(
            direction
        )

        if norm == 0:
            raise ValueError(
                "direction must be non-zero"
            )

        self.direction = (
            direction / norm
        )

    def set_wavelength(
        self,
        wavelength,
    ):
        self.wavelength = float(
            wavelength
        )

    def get_beam_parameters(
        self,
    ):
        return {
            "position": self.position.copy(),
            "direction": self.direction.copy(),
            "wavelength": self.wavelength,
        }

    # ------------------------------------------------

    def emit(self):

        return Ray(
            origin=self.position,
            direction=self.direction,
            wavelength=self.wavelength,
            complex_amplitude=1.0 + 0.0j,
            branch_id="root",
        )
