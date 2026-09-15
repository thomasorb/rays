from __future__ import annotations

from tracing.tracer import Tracer
from tracing.ray import Ray
from tracing.ray_bundle import RayBundle

import numpy as np

class OpticalSystem:
    """
    High-level optical system container.

    Manages:

    - optical elements
    - detectors
    - sources
    - ray tracing
    - wavefront error randomization

    Future:
    - mechanical tolerancing
    - Monte-Carlo
    - ray bundles
    """

    def __init__(self):

        self.tracer = Tracer()

        self.elements = []

        self.detectors = []

        self.source = None

        self.last_rays = []
        
    # ==================================================
    # Source management
    # ==================================================

    def set_source(
        self,
        source,
    ):
        self.source = source

    # ==================================================
    # Element management
    # ==================================================

    def add_element(
        self,
        element,
    ):
        """
        Add any optical element.
        """

        self.elements.append(
            element
        )

        self.tracer.add_element(
            element
        )

    # --------------------------------------------------

    def add_detector(
        self,
        detector,
    ):
        """
        Add detector and register it.

        Detectors are also optical elements.
        """

        self.detectors.append(
            detector
        )

        self.tracer.add_element(
            detector
        )

    # ==================================================
    # Utility
    # ==================================================

    def clear_detectors(
        self,
    ):
        """
        Clear all detector hits.
        """

        for detector in self.detectors:

            detector.clear()

    # --------------------------------------------------

    def clear(
        self,
    ):
        """
        Reset simulation state.
        """

        self.clear_detectors()

    # ==================================================
    # Wavefront errors
    # ==================================================

    def randomize_wavefront_errors(
        self,
    ):
        """
        Generate a new realization of every
        wavefront error object.
        """

        for element in self.elements:

            if hasattr(
                element,
                "randomize_wavefront_error",
            ):

                element.randomize_wavefront_error()

    # ==================================================
    # Trace
    # ==================================================

    def trace(
            self,
            verbose=False,
            progress_callback=None,
    ):
        if self.source is None:

            raise RuntimeError(
                "No source defined."
            )

        self.clear()

        emitted = (
            self.source.emit()
        )

        all_rays = []

        #
        # Single ray source
        #

        if isinstance(
            emitted,
            Ray,
        ):

            if progress_callback is not None:
                progress_callback(
                    value=None,
                    message="Tracing ray 1/1",
                    current=1,
                    total=1,
                )

            rays = (
                self.tracer.trace(
                    emitted,
                    progress_callback=
                        progress_callback,
                )
            )

            all_rays.extend(
                rays
            )

        #
        # Bundle source
        #

        elif isinstance(
            emitted,
            RayBundle,
        ):

            if hasattr(
                emitted,
                "__len__",
            ):
                total_rays = len(
                    emitted
                )
            else:
                total_rays = None

            for index, ray in enumerate(
                emitted,
                start=1,
            ):

                if verbose:
                    if total_rays is None:
                        print(
                            f"Tracing ray "
                            f"{index}"
                        )
                    else:
                        print(
                            f"Tracing ray "
                            f"{index}/{total_rays}"
                        )

                def ray_progress(
                    value=None,
                    message=None,
                    current=None,
                    total=None,
                ):
                    if progress_callback is None:
                        return

                    if total_rays is None:
                        progress_callback(
                            value=None,
                            message=(
                                message
                                or
                                f"Tracing ray "
                                f"{index}"
                            ),
                            current=index,
                            total=None,
                        )
                        return

                    if value is None:
                        progress_callback(
                            value=(
                                (index - 1)
                                / total_rays
                            ),
                            message=(
                                message
                                or
                                f"Tracing ray "
                                f"{index}/{total_rays}"
                            ),
                            current=index,
                            total=total_rays,
                        )
                        return

                    progress_callback(
                        value=(
                            (
                                index - 1
                                + value
                            )
                            / total_rays
                        ),
                        message=(
                            message
                            or
                            f"Tracing ray "
                            f"{index}/{total_rays}"
                        ),
                        current=index,
                        total=total_rays,
                    )

                rays = self.tracer.trace(
                    ray,
                    progress_callback=
                        ray_progress,
                )

                all_rays.extend(
                    rays
                )

                if (
                    progress_callback is not None
                    and total_rays is not None
                ):
                    progress_callback(
                        value=index / total_rays,
                        message=(
                            f"Completed ray "
                            f"{index}/{total_rays}"
                        ),
                        current=index,
                        total=total_rays,
                    )
        else:

            raise TypeError(
                f"Unsupported emitted object: "
                f"{type(emitted)}"
            )

        self.last_rays = (
            all_rays
        )

        return all_rays

    # ==================================================
    # Reports
    # ==================================================

    def report(
        self,
    ):

        print()

        print(
            "================================="
        )

        print(
            "      OPTICAL SYSTEM REPORT"
        )

        print(
            "================================="
        )

        print()

        print(
            f"Elements  : "
            f"{len(self.elements)}"
        )

        print(
            f"Detectors : "
            f"{len(self.detectors)}"
        )

        print()

        for detector in self.detectors:

            detector.report()

    # ==================================================
    # Diagnostics
    # ==================================================

    @property
    def total_intensity(
        self,
    ):
        """
        Sum of coherent detector intensities.
        """

        return sum(
            detector.intensity
            for detector in self.detectors
        )

    # --------------------------------------------------

    @property
    def total_power(
        self,
    ):
        """
        Sum of incoherent powers.
        """

        return sum(
            detector.power
            for detector in self.detectors
        )

    @property
    def n_rays(
        self,
    ):

        return len(
            self.last_rays
        )

    # --------------------------------------------------

    def detector_by_name(
        self,
        name,
    ):

        for detector in self.detectors:

            if detector.name == name:

                return detector

        raise KeyError(
            f"No detector named '{name}'"
        )


    def run_monte_carlo(
        self,
        n_iter,
        detector1,
        detector2,
        progress_callback=None,
    ):

        det1 = []
        det2 = []

        for index in range(n_iter):
            self.randomize_wavefront_errors()

            self.trace(
                progress_callback=
                    progress_callback
            )

            if progress_callback is not None:
                progress_callback(
                    value=(
                        (index + 1)
                        / n_iter
                    ),
                    message=(
                        f"Completed Monte-Carlo "
                        f"{index + 1}/{n_iter}"
                    ),
                    current=index + 1,
                    total=n_iter,
                )

            det1.append(
                detector1.intensity
            )

            det2.append(
                detector2.intensity
            )

        return (
            np.asarray(det1),
            np.asarray(det2),
        )
