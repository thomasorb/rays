import numpy as np
from matplotlib.path import Path as MplPath
from plotting.views import VIEW_MAP

from .utils import (
    _snap_drag_coordinate,
    _material_refractive_index,
    _build_material,
    _rotation_to_euler,
    _normalized_vector,
    _rotation_from_degrees,
    _project_point,
    _project_vector,
    _polyline_distances,
    _point_segment_distance,
)


def _projection_distance(
    projected: dict,
    x: float,
    y: float,
) -> float:
    point = np.array([x, y])

    if projected["kind"] == "volume":
        for face in projected["faces"]:
            if MplPath(face).contains_point(
                point
            ):
                return 0.0

        distances = []
        for face in projected["faces"]:
            distances.extend(
                _polyline_distances(
                    face,
                    point,
                )
            )
        for edge in projected["edges"]:
            distances.extend(
                _polyline_distances(
                    edge,
                    point,
                )
            )
        return min(distances)

    if projected["kind"] == "multi_outline":

        distances = []

        for outline in projected["outlines"]:

            if MplPath(
                outline
            ).contains_point(
                point
            ):
                return 0.0

            distances.extend(
                _polyline_distances(
                    outline,
                    point,
                )
            )

        return min(
            distances
        )

    outline = projected["outline"]
    if MplPath(outline).contains_point(
        point
    ):
        return 0.0

    return min(
        _polyline_distances(
            outline,
            point,
        )
    )




def _element_projection(
    element,
    view: str,
):
    """
    Build picking geometry directly from the
    real optical object.

    Supports:

    - planar optics (Mirror, Detector, Interface)
    - compound optics (Window, Compensator, PlateBeamSplitter)
    - future CornerCube support
    """

    i, j = VIEW_MAP[view]

    if hasattr(
        element,
        "get_outlines",
    ):

        outlines = [
            outline[:, [i, j]]
            for outline in element.get_outlines()
        ]

        points = []

        centers = []

        for outline in element.get_outlines():

            points.extend(
                outline[:, [i, j]]
            )

            centers.append(
                np.mean(
                    outline[:-1],
                    axis=0,
                )
            )

        center = np.mean(
            centers,
            axis=0,
        )

        return {
            "kind": "multi_outline",
            "outlines": outlines,
            "center": center[[i, j]],
            "points": points,
        }
    
    #
    # --------------------------------------------------
    # Compound optics:
    #
    # Window
    # Compensator
    # PlateBeamSplitter
    # --------------------------------------------------
    #

    if (
        hasattr(element, "front")
        and hasattr(element, "back")
        and hasattr(element.front, "get_outline")
        and hasattr(element.back, "get_outline")
    ):

        front = (
            element.front.get_outline()
        )

        back = (
            element.back.get_outline()
        )

        front_xy = front[:, [i, j]]
        back_xy = back[:, [i, j]]

        edges = []

        for p_front, p_back in zip(
            front[:-1],
            back[:-1],
        ):

            edges.append(
                np.array([
                    p_front[[i, j]],
                    p_back[[i, j]],
                ])
            )

        points = (
            list(front_xy)
            + list(back_xy)
        )

        center = (
            np.asarray(
                element.position
            )[[i, j]]
        )

        return {

            "kind": "volume",

            "faces": [
                front_xy,
                back_xy,
            ],

            "edges": edges,

            "center": center,

            "points": points,
        }

    #
    # --------------------------------------------------
    # Volume element
    # --------------------------------------------------
    #

    if hasattr(
        element,
        "get_volume_vertices",
    ):

        vertices = (
            element.get_volume_vertices()
        )

        front = vertices[
            [0, 1, 2, 3, 0]
        ]

        back = vertices[
            [4, 5, 6, 7, 4]
        ]

        edges = []

        for a, b in [
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]:

            edges.append(
                vertices[[a, b]][:, [i, j]]
            )

        return {

            "kind": "volume",

            "faces": [
                front[:, [i, j]],
                back[:, [i, j]],
            ],

            "edges": edges,

            "center":
                vertices.mean(
                    axis=0
                )[[i, j]],

            "points":
                list(vertices[:, [i, j]]),
        }

    #
    # --------------------------------------------------
    # Planar element
    # --------------------------------------------------
    #

    if hasattr(
        element,
        "get_outline",
    ):

        outline = (
            element.get_outline()
        )

        return {

            "kind": "outline",

            "outline":
                outline[:, [i, j]],

            "center":
                element.transform.position[[i, j]],

            "points":
                list(outline[:, [i, j]]),
        }

    #
    # Unsupported object
    #

    return None

def _projected_points(
    element,
    view,
):
    """
    Return all projected 2D points
    used by a projected optical object.
    """

    projected = _element_projection(
        element,
        view,
    )

    if projected is None:
        return []

    return projected["points"]
