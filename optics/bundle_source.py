from __future__ import annotations

import numpy as np

from tracing.ray import Ray
from tracing.ray_bundle import RayBundle
from .base import Optic


class CircularBundleSource(
    Optic
):

    def __init__(
        self,
        position,
        direction,
        wavelength,
        radius,
        n_rays,
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
        self.set_radius(
            radius
        )
        self.set_n_rays(
            n_rays
        )

    # ----------------------------------

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

    def set_radius(
        self,
        radius,
    ):
        radius = float(
            radius
        )

        if radius < 0:
            raise ValueError(
                "radius must be >= 0"
            )

        self.radius = radius

    def set_n_rays(
        self,
        n_rays,
    ):
        n_rays = int(n_rays)

        if n_rays <= 0:
            raise ValueError(
                "n_rays must be > 0"
            )

        self.n_rays = n_rays

    def get_beam_parameters(
        self,
    ):
        return {
            "position": self.position.copy(),
            "direction": self.direction.copy(),
            "wavelength": self.wavelength,
            "radius": self.radius,
            "n_rays": self.n_rays,
        }

    # ----------------------------------

    def emit(
        self,
    ):

        rays = []

        #
        # Normalize bundle power
        #
        amplitude = (
            1.0
            / np.sqrt(self.n_rays)
        )

        #
        # Optical axis
        #
        k = (
            self.direction
            / np.linalg.norm(
                self.direction
            )
        )

        #
        # Build a local orthonormal basis
        # transverse to propagation
        #

        tmp = np.array(
            [0.0, 0.0, 1.0]
        )

        if abs(
            np.dot(k, tmp)
        ) > 0.95:

            tmp = np.array(
                [0.0, 1.0, 0.0]
            )

        u = np.cross(
            k,
            tmp,
        )

        u /= np.linalg.norm(
            u
        )

        v = np.cross(
            k,
            u,
        )

        v /= np.linalg.norm(
            v
        )

        #
        # Generate rays
        #

        for i in range(
            self.n_rays
        ):

            #
            # Uniform disk sampling
            #

            r = (
                self.radius
                * np.sqrt(
                    np.random.rand()
                )
            )

            theta = (
                2.0 * np.pi
                * np.random.rand()
            )

            du = (
                r * np.cos(theta)
            )

            dv = (
                r * np.sin(theta)
            )

            #
            # Position in transverse plane
            #

            origin = (
                self.position
                + du * u
                + dv * v
            )

            rays.append(
                Ray(
                    origin=origin,

                    direction=self.direction,

                    wavelength=self.wavelength,

                    complex_amplitude=
                        amplitude + 0j,

                    branch_id=
                        f"root.{i}",
                )
            )

        return RayBundle(
            rays
        )
