from __future__ import annotations

import numpy as np

from scipy.spatial.transform import Rotation

from .mirror import Mirror


class CornerCube:
    """
    Corner cube retroreflector.

    The corner vertex is located at position.

    optical_axis defines the viewing direction
    of the cube.

    Default:
        optical_axis = [1,0,0]

    Internal geometry:

        three mutually orthogonal mirrors

    whose bisector points along optical_axis.
    """

    def __init__(

        self,

        name,

        position,

        size,

        optical_axis=(1,0,0),
    ):

        self.name = name

        self.size = float(size)

        self._position = np.asarray(
            position,
            dtype=float,
        )

        #
        # ------------------------------------
        # Optical axis
        # ------------------------------------
        #

        optical_axis = np.asarray(
            optical_axis,
            dtype=float,
        )

        optical_axis /= np.linalg.norm(
            optical_axis
        )

        self.optical_axis = (
            optical_axis
        )

        #
        # ------------------------------------
        # Local corner cube axis
        #
        # Bisector of the three reflecting
        # planes.
        # ------------------------------------
        #

        local_axis = np.array(
            [1,1,1],
            dtype=float,
        )

        local_axis /= np.linalg.norm(
            local_axis
        )

        #
        # ------------------------------------
        # Rotation aligning cube axis to
        # requested optical axis.
        # ------------------------------------
        #

        rotation, _ = Rotation.align_vectors(
            [optical_axis],
            [local_axis],
        )

        self.rotation = rotation

        #
        # ------------------------------------
        # Face centres in local coordinates
        # ------------------------------------
        #

        s = self.size / 2

        p1_local = np.array(
            [s,0,0]
        )

        p2_local = np.array(
            [0,s,0]
        )

        p3_local = np.array(
            [0,0,s]
        )

        #
        # ------------------------------------
        # Face centres in world frame
        # ------------------------------------
        #

        p1 = (
            rotation.apply(
                p1_local
            )
            +
            self._position
        )

        p2 = (
            rotation.apply(
                p2_local
            )
            +
            self._position
        )

        p3 = (
            rotation.apply(
                p3_local
            )
            +
            self._position
        )

        #
        # ------------------------------------
        # Mirror orientations
        #
        # Local normals:
        #
        # +X
        # +Y
        # +Z
        # ------------------------------------
        #

        rot_x = (
            rotation
            *
            Rotation.from_euler(
                "y",
                90,
                degrees=True,
            )
        )

        rot_y = (
            rotation
            *
            Rotation.from_euler(
                "x",
                -90,
                degrees=True,
            )
        )

        rot_z = rotation

        #
        # ------------------------------------
        # Mirrors
        # ------------------------------------
        #

        self.m1 = Mirror(

            name=f"{name}.M1",

            position=p1,

            rotation=rot_x,

            width=self.size,
            height=self.size,
        )

        self.m2 = Mirror(

            name=f"{name}.M2",

            position=p2,

            rotation=rot_y,

            width=self.size,
            height=self.size,
        )

        self.m3 = Mirror(

            name=f"{name}.M3",

            position=p3,

            rotation=rot_z,

            width=self.size,
            height=self.size,
        )

    # ==================================================
    # POSITION
    # ==================================================

    @property
    def position(
        self,
    ):

        return self._position

    @position.setter
    def position(
        self,
        new_position,
    ):

        new_position = np.asarray(
            new_position,
            dtype=float,
        )

        delta = (
            new_position
            - self._position
        )

        self.translate(
            delta
        )

    # ==================================================
    # TRANSLATION
    # ==================================================

    def translate(
        self,
        delta,
    ):

        delta = np.asarray(
            delta,
            dtype=float,
        )

        self._position += delta

        self.m1.transform.position += delta
        self.m2.transform.position += delta
        self.m3.transform.position += delta

    # ==================================================
    # SYSTEM
    # ==================================================

    def add_to_system(
        self,
        system,
    ):

        system.add_element(
            self.m1
        )

        system.add_element(
            self.m2
        )

        system.add_element(
            self.m3
        )

    # ==================================================
    # UTILITIES
    # ==================================================

    @property
    def mirrors(
        self,
    ):

        return [
            self.m1,
            self.m2,
            self.m3,
        ]

    # --------------------------------------------------

    def __repr__(
        self,
    ):

        return (
            f"CornerCube("
            f"name={self.name}, "
            f"position={self.position}, "
            f"axis={self.optical_axis}, "
            f"size={self.size}"
            f")"
        )

    def get_outlines(
        self,
    ):
        return [
            self.m1.get_outline(),
            self.m2.get_outline(),
            self.m3.get_outline(),
        ]
