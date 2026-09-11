from __future__ import annotations

import numpy as np

from collections import defaultdict

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
    # Grouping
    # ===================================================

    @property
    def hit_groups(self):
        """
        Group hits by bundle sample.

        Example:

            root.17.R.R
            root.17.T.T

        become

            root.17
        """

        groups = defaultdict(list)

        for hit in self.hits:

            tokens = (
                hit["branch"]
                .split(".")
            )

            if len(tokens) >= 2:

                bundle_id = (
                    tokens[0]
                    + "."
                    + tokens[1]
                )

            else:

                bundle_id = tokens[0]

            groups[bundle_id].append(
                hit
            )

        return groups

    # ===================================================
    # Fields
    # ===================================================

    @property
    def field(self):
        """
        Total coherent field.
        Mostly for debugging.
        """

        return sum(
            hit["field"]
            for hit in self.hits
        )

    # ---------------------------------------------------

    @property
    def incoherent_field_norm(
        self,
    ):

        return sum(
            abs(hit["field"])
            for hit in self.hits
        )

    # ===================================================
    # Physics
    # ===================================================

    @property
    def intensity(self):
        """
        Physical detector intensity.

        For each bundle sample:

            E = sum(fields)

        Then:

            I += |E|²
        """

        intensity = 0.0

        for hits in (
            self.hit_groups.values()
        ):

            E = sum(
                hit["field"]
                for hit in hits
            )

            intensity += (
                abs(E)**2
            )

        return intensity

    # ---------------------------------------------------

    @property
    def power(self):
        """
        Incoherent power.
        Useful for diagnostics.
        """

        return sum(
            abs(hit["field"])**2
            for hit in self.hits
        )

    # ---------------------------------------------------

    @property
    def visibility(self):

        coherent = 0.0
        incoherent = 0.0

        for hits in (
            self.hit_groups.values()
        ):

            E = sum(
                hit["field"]
                for hit in hits
            )

            coherent += abs(E)

            incoherent += sum(
                abs(hit["field"])
                for hit in hits
            )

        if incoherent == 0:
            return 0.0

        return (
            coherent
            / incoherent
        )

    # ===================================================
    # Statistics
    # ===================================================

    @property
    def phases(self):

        return np.array([
            np.angle(
                hit["field"]
            )
            for hit in self.hits
        ])

    # ---------------------------------------------------

    @property
    def n_bundle_samples(self):

        return len(
            self.hit_groups
        )

    # ---------------------------------------------------

    @property
    def mean_intensity_per_sample(
        self,
    ):

        if (
            self.n_bundle_samples
            == 0
        ):
            return 0.0

        return (
            self.intensity
            / self.n_bundle_samples
        )

    # ===================================================
    # Utilities
    # ===================================================

    def clear(
        self,
    ):

        self.hits.clear()

    # ===================================================
    # Reporting
    # ===================================================

    def report(
        self,
        verbose=False,
    ):

        print()

        print(
            f"===== DETECTOR {self.name} ====="
        )

        print(
            f"Hits       : "
            f"{len(self.hits)}"
        )

        print(
            f"Bundle samples : "
            f"{self.n_bundle_samples}"
        )

        print(
            f"Power      : "
            f"{self.power:.6f}"
        )

        print(
            f"Intensity  : "
            f"{self.intensity:.6f}"
        )

        print(
            f"Mean/sample: "
            f"{self.mean_intensity_per_sample:.6f}"
        )

        print(
            f"Phase std  : "
            f"{np.std(self.phases):.6f}"
        )

        print(
            f"Visibility : "
            f"{self.visibility:.6f}"
        )

        if not verbose:
            return

        print()

        for hit in self.hits:

            amp = abs(
                hit["field"]
            )

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
