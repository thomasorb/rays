from __future__ import annotations

import numpy as np

from .planar import PlanarElement

from collections import defaultdict

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

    @property
    def field_groups(self):
        """
        Group fields by bundle sample.

        Example:

            root.0.R.R
            root.0.T.T

        become:

            group 'root.0'
        """

        groups = defaultdict(list)

        for hit in self.hits:

            tokens = hit["branch"].split(".")

            #
            # root.17.R.T
            #
            if len(tokens) >= 2:

                bundle_id = (
                    tokens[0]
                    + "."
                    + tokens[1]
                )

            else:

                bundle_id = tokens[0]

            groups[bundle_id].append(
                hit["field"]
            )

        return groups

    @property
    def incoherent_field_norm(
        self,
    ):
        """
        Sum of field magnitudes.
        """

        return sum(
            abs(hit["field"])
            for hit in self.hits
        )

    # ---------------------------------------------------

    @property
    def intensity(self):
        """
        Physical detector intensity.

        Interfere fields within each
        bundle sample.

        Then sum intensities.
        """

        intensity = 0.0

        for fields in (
            self.field_groups.values()
        ):

            E = sum(fields)

            intensity += abs(E)**2

        return intensity

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

    @property
    def phases(self):

        return np.array([
            np.angle(
                hit["field"]
            )
            for hit in self.hits
        ])

    @property
    def visibility(self):

        coherent = 0.0
        incoherent = 0.0

        for fields in (
            self.field_groups.values()
        ):

            E = sum(fields)

            coherent += abs(E)

            incoherent += sum(
                abs(f)
                for f in fields
            )

        if incoherent == 0:
            return 0.0

        return (
            coherent
            / incoherent
        )
    
    @property
    def n_bundle_samples(self):
        
        return len(
            self.field_groups
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

    def report(
        self,
        verbose=False,
    ):

        print()
        print(
            f"===== DETECTOR {self.name} ====="
        )

        print(
            f"Hits       : {len(self.hits)}"
        )

        print(
            "Bundle samples :",
            self.n_bundle_samples
        )
        
        print(
            f"Power      : {self.power:.6f}"
        )

        print(
            f"Intensity  : {self.intensity:.6f}"
        )

        print(
            "Phase std :",
            np.std(
                self.phases
            )
        )

        print(
            f"Visibility : "
            f"{self.visibility:.6f}"
        )

        if not verbose:
            return

        print()

        for hit in self.hits:

            amp = abs(hit["field"])

            phase = np.angle(
                hit["field"]
            )

            print(
                hit["branch"]
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
