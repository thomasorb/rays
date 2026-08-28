from __future__ import annotations


class RayBundle:

    def __init__(
        self,
        rays,
    ):
        self.rays = list(rays)

    # ----------------------------------

    def __iter__(self):
        return iter(self.rays)

    # ----------------------------------

    def __len__(self):
        return len(self.rays)

    # ----------------------------------

    def append(
        self,
        ray,
    ):
        self.rays.append(ray)
