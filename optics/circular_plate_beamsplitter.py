from __future__ import annotations

import numpy as np

from materials.air import AIR

from .circular_optical_interface import (
    CircularOpticalInterface,
)

import config

class CircularPlateBeamSplitter:
    """
    Circular thick beam splitter.

    The reference position corresponds
    to the coated front surface.

    Geometry:

        AIR
          |
      front interface
          |
       material
          |
      back interface
          |
        AIR
    """

    def __init__(

        self,

        name,

        position,
        rotation,

        thickness,

        diameter,

        material,

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

        self.diameter = float(
            diameter
        )

        self.material = material

        #
        # Front coated interface
        #

        self.front = CircularOpticalInterface(

            name=f"{name}.front",

            position=self.position,

            rotation=self.rotation,

            diameter=self.diameter,

            material1=AIR,
            material2=self.material,

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

        self.back = CircularOpticalInterface(

            name=f"{name}.back",

            position=back_position,

            rotation=self.rotation,

            diameter=self.diameter,

            material1=self.material,
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
            f"CircularPlateBeamSplitter("
            f"name={self.name}, "
            f"diameter={self.diameter}, "
            f"thickness={self.thickness}, "
            f"material={self.material.name}"
            f")"
        )
