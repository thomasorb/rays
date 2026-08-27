from __future__ import annotations

import numpy as np


class Tracer:
    """
    Generic ray tracing engine.
    """

    def __init__(self):
        self.elements = []

    def add_element(
        self,
        element,
    ):
        self.elements.append(
            element
        )

    # -------------------------------------------------------------

    def find_next_element(
        self,
        ray,
    ):
        """
        Search nearest intersection.

        Order independent.
        """

        best_distance = np.inf
        best_element = None

        for element in self.elements:

            distance = element.intersect(
                ray
            )

            if distance is None:
                continue

            if distance < best_distance:

                best_distance = distance
                best_element = element

        if best_element is None:
            return None, None

        return (
            best_element,
            best_distance,
        )

    # -------------------------------------------------------------

    def trace(
        self,
        ray,
        max_bounces=100,
    ):
        """
        Trace ray until death.
        """

        for _ in range(max_bounces):

            if not ray.is_alive:
                break

            element, distance = (
                self.find_next_element(
                    ray
                )
            )

            if element is None:
                break

            ray.propagate(distance)

            element.interact(ray)

        return ray
