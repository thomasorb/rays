from __future__ import annotations

import numpy as np


class Ray:
    """
    Optical ray.
    """

    def __init__(
        self,
        origin,
        direction,
        wavelength=632.8e-9,
        amplitude=1.0,
    ):
        self.origin = np.asarray(
            origin,
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

        self.amplitude = amplitude

        self.geometric_length = 0.0

        self.optical_length = 0.0

        self.phase = 0.0

        self.path = [
            self.origin.copy()
        ]

        self.is_alive = True

        self.segments = []
        
    # ------------------------------------------------------------------

    def propagate(
        self,
        distance,
        refractive_index=1.0,
    ):
        """
        Propagate ray and store segment.
        """

        start = self.origin.copy()

        self.origin = (
            self.origin
            + distance * self.direction
        )

        end = self.origin.copy()

        self.segments.append(
            {
                "start": start,
                "end": end,
                "distance": distance,
                "n": refractive_index,
                "opl": (
                    distance
                    * refractive_index
                ),
            }
        )

        self.geometric_length += distance

        self.optical_length += (
            distance
            * refractive_index
        )

        self.phase = (
            2.0
            * np.pi
            * self.optical_length
            / self.wavelength
        )

        self.path.append(
            self.origin.copy()
        )

    @property
    def opl(self):

        return sum(
            seg["opl"]
            for seg in self.segments
        )

    @property
    def geometric_distance(self):

        return sum(
            seg["distance"]
            for seg in self.segments
        )

    def summary(self):

        print()

        print(
            "===== RAY SUMMARY ====="
        )

        for i, segment in enumerate(
            self.segments
        ):

            print(
                f"{i:03d} "
                f"L={segment['distance']:.6f} "
                f"n={segment['n']:.4f} "
                f"OPL={segment['opl']:.6f}"
            )

        print()

        print(
            f"Total geometric length : "
            f"{self.geometric_distance:.6f}"
        )

        print(
            f"Total optical length : "
            f"{self.opl:.6f}"
        )

        print(
            f"Phase : "
            f"{self.phase:.6e}"
        )

    def opd(
        self,
        other_ray,
    ):
        """
        Optical path difference.
        """

        return (
            self.optical_length
            - other_ray.optical_length
        )

    def phase_difference(
        self,
        other_ray,
    ):
        """
        Relative phase.
        """

        opd = self.opd(
            other_ray
        )

        return (
            2*np.pi
            * opd
            / self.wavelength
        )
