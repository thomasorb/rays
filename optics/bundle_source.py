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

        for i in range(
            self.n_rays
        ):

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

            x = r * np.cos(theta)
            y = r * np.sin(theta)

            origin = (
                self.position.copy()
            )

            #
            # NOTE:
            # This is still using the global XY plane.
            # We'll improve this later by constructing
            # a local transverse basis (u,v).
            #
            origin[0] += x
            origin[1] += y

            rays.append(
                Ray(
                    origin=origin,
                    direction=self.direction,
                    wavelength=self.wavelength,

                    #
                    # Normalized complex amplitude
                    #
                    complex_amplitude=
                        amplitude + 0j,

                    branch_id=
                        f"root.{i}",
                )
            )

        return RayBundle(
            rays
        )
