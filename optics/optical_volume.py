from __future__ import annotations

import numpy as np

import logger

import config

class OpticalVolume:
    """
    Base class for thick optical elements.

    Requires:

        self.front
        self.back
    """

    @property
    def interfaces(
        self,
    ):
        return [
            self.front,
            self.back,
        ]

    # --------------------------------------------------


    def register_interfaces(self):

        for interface in self.interfaces:

            interface.parent_volume = self

            logger.log(
                "REGISTERED",
                interface.name,
                "->",
                self.name,
            )

        
    def contains_between_surfaces(
        self,
        point,
    ):
        """
        Point is inside if it lies
        between the front and back planes.
        """

        front_local = (
            self.front.transform
            .world_to_local_point(
                point
            )
        )

        back_local = (
            self.back.transform
            .world_to_local_point(
                point
            )
        )

        return (
            front_local[2] >= -config.GEOMETRY_EPSILON
            and
            back_local[2] <= config.GEOMETRY_EPSILON
        )
