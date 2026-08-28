from __future__ import annotations

import numpy as np

from .planar import PlanarElement


class Detector(
    PlanarElement
):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs
        )

        self.hits = []

    # ===================================================
    # Interaction
    # ===================================================

    def interact(
        self,
       ray,
    ):

        local_point = (
            self.transform
            .world_to_local_point(
                ray.origin
            )
        )

        self.hits.append(
            {
                "branch": ray.branch_id,
                "position": local_point,
                "opl": ray.opl,
                "field": ray.field,
            }
        )

        ray.is_alive = False

        ray.termination_reason = (
            "detector"
        )

        return [ray]

    # ===================================================
    # Field quantities
    # ===================================================

    @property
    def field(self):
        """
        Total complex field.
        """

        return sum(
            hit["field"]
            for hit in self.hits
        )

    # ---------------------------------------------------

    @property
    def intensity(self):
        """
        Coherent intensity.
        """

        return abs(
            self.field
        )**2

    # ---------------------------------------------------

    @property
    def power(self):
        """
        Incoherent sum of powers.
        Useful for debugging.
        """

        return sum(
            abs(hit["field"])**2
            for hit in self.hits
        )

    # ===================================================
    # Utilities
    # ===================================================

    def clear(self):
        """
        Clear detector before a new run.
        """

        self.hits.clear()

    # ---------------------------------------------------

    def report(self):

        print()

        print(
            f"===== DETECTOR "
            f"{self.name} ====="
        )

        print(
            f"Hits       : "
            f"{len(self.hits)}"
        )

        print(
            f"Power      : "
            f"{self.power:.6f}"
        )

        print(
            f"Intensity  : "
            f"{self.intensity:.6f}"
        )

        print()

        print(
            f"Field = "
            f"{self.field.real:.6e}"
            f" + "
            f"{self.field.imag:.6e}j"
        )

        print()

        for hit in self.hits:

            amp = abs(
                hit["field"]
            )

            phase = np.angle(
                hit["field"]
            )

            print(
                f"{hit['branch']}"
            )

            print(
                f"    OPL   : "
                f"{hit['opl']:.6f}"
            )

            print(
                f"    Amp   : "
                f"{amp:.6f}"
            )

            print(
                f"    Phase : "
                f"{phase:.6f}"
            )

            print()
