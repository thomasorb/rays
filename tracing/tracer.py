from __future__ import annotations

import numpy as np

import logger

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
        max_interactions=1000,
    ):

        active_rays = [
            initial_ray
        ]

        finished_rays = []

        interactions = 0

        while active_rays:

            ray = active_rays.pop()

            if interactions > max_interactions:
                logger.log(f'Max interactions ({max_interactions}) reached, terminating ray tracing.')
                break

            interactions += 1

            element, distance = (
                self.find_next_element(
                    ray
                )
            )


            #
            # Volume escape test
            #
            # We must test the NEXT intersection
            # point, not the current ray origin.
            #

            if ray.current_volume is not None:

                if element is None:

                    ray.is_alive = False

                    ray.termination_reason = (
                        "Escaped optical volume"
                    )

                    finished_rays.append(
                        ray
                    )

                    continue

                next_point = (
                    ray.origin
                    +
                    distance
                    * ray.direction
                )

                logger.log(
                    "CHECKING",
                    ray.branch_id,
                    next_point,
                    ray.current_volume.name,
                )

                logger.log(
                    "LOCAL",
                    ray.current_volume.front
                    .transform
                    .world_to_local_point(
                        next_point
                    )
                )

                inside = (
                    ray.current_volume
                    .contains_world_point(
                        next_point
                    )
                )

                logger.log(
                    "NEXT INSIDE:",
                    inside
                )

                if not inside:

                    ray.is_alive = False

                    ray.termination_reason = (
                        "Escaped optical volume"
                    )

                    finished_rays.append(
                        ray
                    )

                    continue
            
            if element is None:

                finished_rays.append(
                    ray
                )

                continue

            logger.log(
                element.name,
                type(element).__name__,
                ray.current_material.name,
                ray.branch_id,
            )

            ray.propagate(distance)

            logger.log(
                "CALLING",
                element.name,
                type(element).__name__,
                element.interact,
            )
            
            new_rays = element.interact(
                ray
            )


            logger.log(
                "RETURNED",
                element.name,
                len(new_rays),
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
