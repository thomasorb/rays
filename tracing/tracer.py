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
        initial_ray,
        max_interactions=100,
    ):

        active_rays = [
            initial_ray
        ]

        finished_rays = []

        interactions = 0

        while active_rays:

            ray = active_rays.pop()

            if interactions > max_interactions:
                break

            interactions += 1

            element, distance = (
                self.find_next_element(
                    ray
                )
            )

            if element is None:

                finished_rays.append(
                    ray
                )

                continue

            ray.propagate(distance)

            new_rays = element.interact(
                ray
            )

            for child in new_rays:

                if child.is_alive:

                    active_rays.append(
                        child
                    )

                else:

                    finished_rays.append(
                        child
                    )

        return finished_rays
