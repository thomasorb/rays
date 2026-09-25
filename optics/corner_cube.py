from __future__ import annotations

import numpy as np
from matplotlib.patches import Circle

from scipy.spatial.transform import Rotation

from .mirror import Mirror


class CornerCube:
    """
    Corner cube retroreflector.

    Parameters
    ----------
    position
        Corner vertex position.

    aperture
        Effective clear aperture.

    optical_axis
        Viewing direction of the cube.

    Notes
    -----
    The public parameter is the clear
    aperture.

    Internal mirror dimensions are
    automatically derived.
    """

    def __init__(

        self,

        name,

        position,

        aperture=None,

        size=None,

        optical_axis=(1, 0, 0),
    ):

        #
        # Backward compatibility
        #

        if aperture is None:

            if size is None:

                raise ValueError(
                    "CornerCube requires "
                    "aperture or size."
                )

            aperture = size

        #
        # Public parameter
        #

        self.aperture = float(
            aperture
        )

        #
        # Internal mirror size
        #
        # For a cubic corner cube:
        #
        # mirror_size = √2 × aperture
        #

        self.mirror_size = (
            np.sqrt(3) / 2
            * self.aperture
        )

        self.name = name

        self._position = np.asarray(
            position,
            dtype=float,
        )

        #
        # Optical axis
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
        # Local cube bisector
        #

        local_axis = np.array(
            [1, 1, 1],
            dtype=float,
        )

        local_axis /= np.linalg.norm(
            local_axis
        )

        rotation, _ = (
            Rotation.align_vectors(
                [optical_axis],
                [local_axis],
            )
        )

        self.rotation = rotation

        #
        # Mirror centres
        #

        s = (
            self.mirror_size / 2
        )

        p1_local = np.array(
            [s, 0, 0]
        )

        p2_local = np.array(
            [0, s, 0]
        )

        p3_local = np.array(
            [0, 0, s]
        )

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
        # Mirror orientations
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
        # Mirrors
        #

        self.m1 = Mirror(

            name=f"{name}.M1",

            position=p1,

            rotation=rot_x,

            width=self.mirror_size,
            height=self.mirror_size,
        )

        self.m2 = Mirror(

            name=f"{name}.M2",

            position=p2,

            rotation=rot_y,

            width=self.mirror_size,
            height=self.mirror_size,
        )

        self.m3 = Mirror(

            name=f"{name}.M3",

            position=p3,

            rotation=rot_z,

            width=self.mirror_size,
            height=self.mirror_size,
        )

    # ==================================================
    # Backward compatibility
    # ==================================================

    @property
    def size(
        self,
    ):
        """
        Backward compatibility.

        Old code may still query:

            cc.size
        """

        return self.mirror_size

    # ==================================================
    # Position
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
    # Translation
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
    # System
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


    def draw_overlay(
        self,
        view,
        ax,
    ):
        """
        Draw projected clear aperture.
        """

        from plotting.views import VIEW_MAP

        i, j = VIEW_MAP[view]

        #
        # Circle radius
        #

        r = self.aperture / 2

        #
        # Plane normal
        #

        n = self.optical_axis

        #
        # Build orthonormal basis
        #

        if abs(n[2]) < 0.9:

            reference = np.array(
                [0, 0, 1]
            )

        else:

            reference = np.array(
                [0, 1, 0]
            )

        u = np.cross(
            n,
            reference,
        )

        u /= np.linalg.norm(
            u
        )

        v = np.cross(
            n,
            u,
        )

        v /= np.linalg.norm(
            v
        )

        #
        # Circle points
        #

        angles = np.linspace(
            0,
            2*np.pi,
            200,
        )

        circle = []

        for t in angles:

            point = (

                self.position

                + r*np.cos(t)*u

                + r*np.sin(t)*v
            )

            circle.append(
                point
            )

        circle = np.asarray(
            circle
        )

        #
        # Project
        #

        ax.plot(

            circle[:, i],

            circle[:, j],

            "--",

            color="cyan",

            linewidth=1.5,

            alpha=0.8,

            zorder=25,
        )

        #
        # Vertex
        #

        ax.plot(

            self.position[i],

            self.position[j],

            marker="+",

            color="cyan",

            markersize=8,

            zorder=30,
        )

    # ==================================================
    # Editor support
    # ==================================================

    def get_outlines(
        self,
    ):

        return [

            self.m1.get_outline(),

            self.m2.get_outline(),

            self.m3.get_outline(),
        ]

    # ==================================================
    # Utilities
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
            f"aperture={self.aperture}, "
            f"mirror_size={self.mirror_size}"
            f")"
        )
