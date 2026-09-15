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

class PlateBeamSplitter(
    Optic
):
    """
    Plate beam splitter.

    Position corresponds to the
    semi-reflective coating.
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

        R=0.5,
        T=0.5,

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

        self.material = material

        #
        # Semi-reflective coating
        #

        self.front = OpticalInterface(

            name=name + ".front",

            position=position,

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
