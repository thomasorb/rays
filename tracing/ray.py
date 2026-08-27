from __future__ import annotations

import copy
import numpy as np


class Ray:

    def __init__(
        self,
        origin,
        direction,
        wavelength=632.8e-9,
        amplitude=1.0,
        branch_id="root",
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

        self.branch_id = branch_id

        self.geometric_length = 0.0
        self.optical_length = 0.0

        self.phase = 0.0

        self.path = [
            self.origin.copy()
        ]

        self.segments = []

        self.is_alive = True

        self.termination_reason = None
        
    # ----------------------------------------------------------

    def clone(self):

        return copy.deepcopy(self)

    # ----------------------------------------------------------

    def propagate(
        self,
        distance,
        refractive_index=1.0,
    ):

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

    # ----------------------------------------------------------

    @property
    def opl(self):

        return sum(
            seg["opl"]
            for seg in self.segments
        )

    # ----------------------------------------------------------

    @property
    def geometric_distance(self):

        return sum(
            seg["distance"]
            for seg in self.segments
        )

    @property
    def generation(self):
        """
        Number of beam splitter generations.
        """

        return self.branch_id.count(".")

    # ----------------------------------------------------------

    def opd(
        self,
        other_ray,
    ):
        return (
            self.optical_length
            - other_ray.optical_length
        )

    # ----------------------------------------------------------

    def phase_difference(
        self,
        other_ray,
    ):
        opd = self.opd(
            other_ray
        )

        return (
            2.0
            * np.pi
            * opd
            / self.wavelength
        )

    # ----------------------------------------------------------

    def summary(self):

        print()
        print(
            f"===== {self.branch_id} ====="
        )

        for i, segment in enumerate(
            self.segments
        ):

            print(
                f"{i:03d}  "
                f"L={segment['distance']:.6f}  "
                f"n={segment['n']:.3f}  "
                f"OPL={segment['opl']:.6f}"
            )

        print()

        print(
            "Total geometric length:",
            self.geometric_distance,
        )

        print(
            "Total optical length:",
            self.opl,
        )

        print(
            "Amplitude:",
            self.amplitude,
        )

        print(
            "Phase:",
            self.phase,
        )
