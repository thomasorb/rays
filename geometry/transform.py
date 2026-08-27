"""
Rigid body transforms.

All optical elements are defined in their own local frame.

The local optical surface is generally defined by:

    z = 0

and transformed to world coordinates through:

    position
    rotation

Author: Thomas's interferometer project
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


class Transform:
    """
    Position + orientation.
    """

    def __init__(
        self,
        position=None,
        rotation=None,
    ):
        self.position = (
            np.zeros(3)
            if position is None
            else np.asarray(position, dtype=float)
        )

        if rotation is None:
            self.rotation = Rotation.identity()
        elif isinstance(rotation, Rotation):
            self.rotation = rotation
        else:
            raise TypeError(
                "rotation must be scipy Rotation"
            )

    # ------------------------------------------------------------------
    # point transforms
    # ------------------------------------------------------------------

    def local_to_world_point(
        self,
        point: np.ndarray,
    ) -> np.ndarray:

        point = np.asarray(point)

        return (
            self.rotation.apply(point)
            + self.position
        )

    def world_to_local_point(
        self,
        point: np.ndarray,
    ) -> np.ndarray:

        point = np.asarray(point)

        return self.rotation.inv().apply(
            point - self.position
        )

    # ------------------------------------------------------------------
    # vector transforms
    # ------------------------------------------------------------------

    def local_to_world_vector(
        self,
        vector: np.ndarray,
    ) -> np.ndarray:

        return self.rotation.apply(vector)

    def world_to_local_vector(
        self,
        vector: np.ndarray,
    ) -> np.ndarray:

        return self.rotation.inv().apply(vector)

    # ------------------------------------------------------------------

    @property
    def normal(self):
        """
        Local +Z axis expressed in world frame.
        """
        return self.rotation.apply(
            np.array([0.0, 0.0, 1.0])
        )
