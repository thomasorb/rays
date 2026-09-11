from __future__ import annotations

import numpy as np

from tracing.ray import Ray
from tracing.ray_bundle import RayBundle


class CircularBundleSource:

    def __init__(
        self,
        position,
        direction,
        wavelength,
        radius,
        n_rays,
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

        self.radius = radius

        self.n_rays = n_rays

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
