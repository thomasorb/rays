from __future__ import annotations

from .planar import PlanarElement


class Detector(
    PlanarElement
):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs
        )

        self.hits = []

        
    # ----------------------------------------------------------
    def interact(
        self,
        ray,
        ):

        local_point = (
            self.transform
            .world_to_local_point(
                ray.origin
            )
        )

        self.hits.append(
            {
                "branch": ray.branch_id,
                "position": local_point,
                "opl": ray.optical_length,
                "phase": ray.phase,
                "amplitude": ray.amplitude,
            }
        )

        ray.is_alive = False

        ray.termination_reason = "detector"
        
        return [ray]
