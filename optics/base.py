from __future__ import annotations

from geometry.transform import Transform
from geometry.plane import Plane


class Optic:
    pass


class OpticalElement(
    Optic
):

    def __init__(
        self,
        position=None,
        rotation=None,
        name=None,
        wavefront_error=None,
    ):
        self.name = name

        self.transform = Transform(
            position=position,
            rotation=rotation,
        )

        self.surface = Plane(
            self.transform
        )

        self.wavefront_error = (
            wavefront_error
        )

    # --------------------------------------------------

    @property
    def normal(self):
        return self.transform.normal

    # --------------------------------------------------

    def randomize_wavefront_error(
        self,
    ):

        if (
            self.wavefront_error
            is not None
        ):
            self.wavefront_error.randomize()

    # --------------------------------------------------

    def apply_wavefront_error(
        self,
        ray,
    ):

        if self.wavefront_error is None:
            return

        local_point = (
            self.transform
            .world_to_local_point(
                ray.origin
            )
        )

        x = local_point[0]
        y = local_point[1]

        opd_error = (
            self.wavefront_error.opd(
                x,
                y,
            )
        )

        ray.optical_length += (
            opd_error
        )

        ray.phase_errors.append(
            {
                "element": self.name,
                "opd_error": opd_error,
            }
        )

    # --------------------------------------------------

    def intersect(
        self,
        ray,
    ):
        return self.surface.intersect(
            ray.origin,
            ray.direction,
        )

    # --------------------------------------------------

    def interact(
        self,
        ray,
    ):
        raise NotImplementedError
