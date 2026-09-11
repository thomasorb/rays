from __future__ import annotations


class Material:
    """
    Base class for optical materials.
    """

    name = "Material"

    def n(
        self,
        wavelength,
    ):
        raise NotImplementedError


    def __repr__(self):

        return self.name
