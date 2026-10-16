from __future__ import annotations

import numpy as np

import logger
import config


class BaseOpticalInterface:

    def __init__(

        self,

        material1,
        material2,

        R=config.DEFAULT_R,
        T=config.DEFAULT_T,

        phase_reflection=np.pi / 2,

        max_generation_depth=config.MAX_RAY_GENERATION_DEPTH,

        *args,
        **kwargs,
    ):

        super().__init__(
            *args,
            **kwargs,
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

        logger.log(
            "INTERACT",
            self.name,
            "material=",
            ray.current_material.name,
            "volume=",
            ray.current_volume,
        )

        if (
            ray.generation
            >= self.max_generation_depth
        ):
            return [ray]

        normal = (
            self.normal.copy()
        )

        if np.dot(
            ray.direction,
            normal,
        ) > 0:
            normal *= -1.0

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
                f"reached interface {self} "
                f"with pair "
                f"({self.material1.name}, "
                f"{self.material2.name}) "
                f"on branch {ray.branch_id}"
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

            reflected = ray.clone()

            reflected.branch_id += ".R"
            reflected.generation += 1

            d = reflected.direction

            reflected.direction = (

                d

                - 2.0
                * np.dot(d, normal)
                * normal
            )

            reflected.direction /= (
                np.linalg.norm(
                    reflected.direction
                )
            )

            reflected.origin += (
                config.GEOMETRY_EPSILON
                * reflected.direction
            )

            reflected.complex_amplitude *= (

                np.sqrt(self.R)

                * np.exp(
                    1j
                    * self.phase_reflection
                )
            )

            reflected.current_material = (
                material_in
            )

            reflected.current_volume = (
                ray.current_volume
            )


            logger.log(
                "REFLECTED",
                self.name,
                reflected.branch_id,
                reflected.current_material.name,
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
                    np.linalg.norm(dT)
                )

                transmitted.origin += (
                    config.GEOMETRY_EPSILON
                    * transmitted.direction
                )

                transmitted.complex_amplitude *= (
                    np.sqrt(self.T)
                )

                transmitted.current_material = (
                    material_out
                )

                volume = getattr(
                    self,
                    "parent_volume",
                    None,
                )

                if volume is not None:

                    #
                    # Outside -> inside
                    #

                    if ray.current_volume is None:

                        transmitted.current_volume = (
                            volume
                        )

                    #
                    # Inside -> outside
                    #

                    else:

                        transmitted.current_volume = (
                            None
                        )

                logger.log(
                    "TRANSMITTED",
                    self.name,
                    transmitted.branch_id,
                    transmitted.current_material.name,
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
