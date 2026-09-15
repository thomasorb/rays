from __future__ import annotations

import numpy as np

from materials.constant_index import (
    ConstantIndex
)

from .base import Optic
from .optical_interface import (
    OpticalInterface
)

from materials.air import AIR

class Window(
    Optic
):
    """
    Plane-parallel plate.

    The position corresponds
    to the front surface.
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

        front_R=0.0,
        front_T=1.0,

        back_R=0.0,
        back_T=1.0,
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

        #
        # Front interface
        #

        self.front = OpticalInterface(

            name=name + ".front",

            position=position,

            rotation=rotation,

            width=width,
            height=height,

            material1=AIR,
            material2=material,

            R=front_R,
            T=front_T,
        )

        #
        # Back interface
        #

        normal = (
            self.front.normal
        )

        back_position = (
            self.position
            + self.thickness
            * normal
        )

        self.back = OpticalInterface(

            name=name + ".back",

            position=back_position,

            rotation=rotation,

            width=width,
            height=height,

            material1=material,
            material2=AIR,

            R=back_R,
            T=back_T,
        )

    # ---------------------------------

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

    # ---------------------------------

    @property
    def interfaces(
        self,
    ):
        return [
            self.front,
            self.back,
        ]

    # ---------------------------------

    @property
    def normal(
        self,
    ):
        return self.front.normal
