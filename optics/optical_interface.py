from __future__ import annotations

import numpy as np

from .planar import PlanarElement


class OpticalInterface(
    PlanarElement
):

    def __init__(
        self,

        material1,
        material2,

        R=0.0,
        T=1.0,

        phase_reflection=np.pi / 2,

        max_generation_depth=10,

        *args,
        **kwargs,
    ):

        super().__init__(
            *args,
            **kwargs
        )

        self.material1 = material1
        self.material2 = material2

        self.R = float(R)
        self.T = float(T)

        self.phase_reflection = (
            phase_reflection
        )

        self.max_generation_depth = (
            max_generation_depth
        )

    # ==================================================
    # Snell
    # ==================================================

    def refract(
        self,
        direction,
        normal,
        n_in,
        n_out,
    ):

        cos_i = (
            -np.dot(
                normal,
                direction,
            )
        )

        eta = (
            n_in / n_out
        )

        k = (
            1.0
            -
            eta**2
            * (
                1.0
                - cos_i**2
            )
        )

        #
        # Total internal reflection
        #

        if k < 0.0:

            return None

        return (
            eta * direction
            +
            (
                eta * cos_i
                - np.sqrt(k)
            )
            * normal
        )

    # ==================================================
    # Interaction
    # ==================================================

    def interact(
        self,
        ray,
    ):

        if (
            ray.generation
            >= self.max_generation_depth
        ):
            return [ray]

        normal = (
            self.normal.copy()
        )

        #
        # Orient normal
        #

        if np.dot(
            ray.direction,
            normal,
        ) > 0:

            normal *= -1.0

        #
        # Determine side
        #

        material_in = (
            ray.current_material
        )

        if material_in is self.material1:

            material_out = (
                self.material2
            )

        elif material_in is self.material2:

            material_out = (
                self.material1
            )

        else:

            raise RuntimeError(
                f"Ray material "
                f"{material_in.name} "
                f"is not part of "
                f"interface "
                f"({self.material1.name}, "
                f"{self.material2.name})"
            )

        n_in = material_in.n(
            ray.wavelength
        )

        n_out = material_out.n(
            ray.wavelength
        )

        rays = []

        #
        # Reflection
        #

        if self.R > 0:

            reflected = (
                ray.clone()
            )

            reflected.branch_id += (
                ".R"
            )

            reflected.generation += 1

            d = reflected.direction

            reflected.direction = (
                d
                -
                2.0
                * np.dot(
                    d,
                    normal
                )
                * normal
            )

            reflected.direction /= (
                np.linalg.norm(
                    reflected.direction
                )
            )

            reflected.complex_amplitude *= (
                np.sqrt(self.R)
                *
                np.exp(
                    1j
                    * self.phase_reflection
                )
            )

            reflected.current_material = (
                material_in
            )

            rays.append(
                reflected
            )

        #
        # Transmission
        #

        if self.T > 0:

            dT = self.refract(
                ray.direction,
                normal,
                n_in,
                n_out,
            )

            if dT is not None:

                transmitted = (
                    ray.clone()
                )

                transmitted.branch_id += (
                    ".T"
                )

                transmitted.generation += 1

                transmitted.direction = (
                    dT
                    /
                    np.linalg.norm(
                        dT
                    )
                )

                transmitted.complex_amplitude *= (
                    np.sqrt(
                        self.T
                    )
                )

                transmitted.current_material = (
                    material_out
                )

                rays.append(
                    transmitted
                )

        ray.is_alive = False

        return rays

    # ==================================================
    # Display
    # ==================================================

    def __repr__(
        self,
    ):

        return (
            f"{self.__class__.__name__}"
            f"("
            f"{self.material1.name}"
            f" -> "
            f"{self.material2.name}"
            f", "
            f"R={self.R}, "
            f"T={self.T}"
            f")"
        )

