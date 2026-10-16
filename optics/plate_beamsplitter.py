from __future__ import annotations

import numpy as np

from scipy.spatial.transform import Rotation

from materials.air import AIR

from .optical_interface import (
    OpticalInterface
)

import config


class PlateBeamSplitter:
    """
    Thick plate beamsplitter.

    The reference position corresponds
    to the semi-reflective front coating.

    A wedge can optionally be applied
    to the rear surface.
    """

    def __init__(

        self,

        name,

        position,
        rotation,

        thickness,

        width,
        height,

        material,

        wedge_angle_deg=0.0,

        R=config.DEFAULT_BEAMSPLITTER_R,
        T=config.DEFAULT_BEAMSPLITTER_T,

        back_R=config.DEFAULT_R,
        back_T=config.DEFAULT_T,
    ):

        self.name = name

        self.position = np.asarray(
            position,
            dtype=float,
        )

        self.rotation = rotation

        self.thickness = float(
            thickness
        )

        self.width = width
        self.height = height

        self.material = material

        self.wedge_angle_deg = float(
            wedge_angle_deg
        )

        #
        # Front coated interface
        #

        self.front = OpticalInterface(

            name=name + ".front",

            position=self.position,

            rotation=rotation,

            width=width,
            height=height,

            material1=AIR,
            material2=material,

            R=R,
            T=T,
        )

        #
        # Rear interface
        #

        normal = (
            self.front.normal
        )

        back_position = (

            self.position

            + self.thickness
            * normal
        )

        #
        # Apply wedge
        #
        # Rotation about local Y.
        #

        back_rotation = (

            rotation

            * Rotation.from_euler(
                "y",
                self.wedge_angle_deg,
                degrees=True,
            )
        )

        self.back = OpticalInterface(

            name=name + ".back",

            position=back_position,

            rotation=back_rotation,

            width=width,
            height=height,

            material1=material,
            material2=AIR,

            R=back_R,
            T=back_T,
        )

    # ==================================================
    # System
    # ==================================================

    def add_to_system(
        self,
        system,
    ):

        system.add_element(
            self.front
        )

        system.add_element(
            self.back
        )

    # ==================================================
    # Display
    # ==================================================

    def get_outlines(
        self,
    ):

        return [

            self.front.get_outline(),

            self.back.get_outline(),
        ]

    # ==================================================
    # Utilities
    # ==================================================

    @property
    def interfaces(
        self,
    ):

        return [
            self.front,
            self.back,
        ]

    @property
    def normal(
        self,
    ):

        return self.front.normal

    # --------------------------------------------------

    def __repr__(
        self,
    ):

        return (
            f"PlateBeamSplitter("
            f"name={self.name}, "
            f"width={self.width}, "
            f"height={self.height}, "
            f"thickness={self.thickness}, "
            f"wedge_angle_deg={self.wedge_angle_deg}, "
            f"material={self.material.name}"
            f")"
        )

