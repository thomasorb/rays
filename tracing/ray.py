from __future__ import annotations

import copy
import numpy as np
from materials.air import AIR


class Ray:

    def __init__(
        self,
        origin,
        direction,
        wavelength=632.8e-9,
        complex_amplitude=1.0 + 0.0j,
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

        #
        # Complex field amplitude
        #
        self.complex_amplitude = (
            complex(complex_amplitude)
        )

        self.branch_id = branch_id

        self.geometric_length = 0.0
        self.optical_length = 0.0

        self.path = [
            self.origin.copy()
        ]

        self.segments = []

        self.is_alive = True

        self.termination_reason = None

        self.phase_errors = []

        self.generation = 0
  
        self.current_material = AIR
    # ==================================================
    # utilities
    # ==================================================

    def clone(self):

        return copy.deepcopy(self)

    # ==================================================
    # propagation
    # ==================================================
    def propagate(
        self,
        distance,
    ):

        self.origin = (
            self.origin
            + distance * self.direction
        )

        self.path.append(
            self.origin.copy()
        )

        self.geometric_length += (
            distance
        )

        n = self.current_material.n(
            self.wavelength
        )

        self.optical_length += (
            distance * n
        )

    # ==================================================
    # properties
    # ==================================================

    @property
    def amplitude(self):

        return abs(
            self.complex_amplitude
        )

    # --------------------------------------------------

    @property
    def phase(self):

        return (
            2*np.pi
            * self.optical_length
            / self.wavelength
        ) % (2*np.pi)

    # --------------------------------------------------

    @property
    def total_phase(self):

        return (
            2*np.pi
            * self.optical_length
            / self.wavelength
        )

    # --------------------------------------------------

    @property
    def field(self):

        return (
            self.complex_amplitude
            * np.exp(
                1j*self.phase
            )
        )

    # --------------------------------------------------

    @property
    def opl(self):

        return self.optical_length

    # ==================================================
    # interferometry
    # ==================================================

    def opd(
        self,
        other_ray,
    ):
        return (
            self.opl
            - other_ray.opl
        )

    # --------------------------------------------------

    def phase_difference(
        self,
        other_ray,
    ):
        return (
            2*np.pi
            * self.opd(other_ray)
            / self.wavelength
        ) % (2*np.pi)

    # ==================================================
    # reporting
    # ==================================================

    def summary(self):

        print()
        print(
            f"===== {self.branch_id} ====="
        )

        for i, seg in enumerate(
            self.segments
        ):

            print(
                f"{i:03d} "
                f"L={seg['distance']:.6f} "
                f"n={seg['n']:.4f} "
                f"OPL={seg['opl']:.6f}"
            )

        print()

        print(
            "Geometric length :",
            self.geometric_length
        )

        print(
            "Optical length   :",
            self.opl
        )

        print(
            "Amplitude        :",
            self.amplitude
        )

        print(
            "Phase            :",
            self.phase
        )

        print(
            "Complex amplitude:",
            self.complex_amplitude
        )

        print(
            "Field            :",
            self.field
        )

        if self.termination_reason:

            print(
                "Termination      :",
                self.termination_reason
            )
