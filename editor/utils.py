import warnings
import numpy as np
from scipy.spatial.transform import Rotation
from materials.air import AIR
from materials.bk7 import BK7
from materials.constant_index import ConstantIndex
from materials.fused_silica import FusedSilica
from plotting.views import VIEW_MAP


def _rotation_from_degrees(
    rotation_deg,
) -> Rotation:
    return Rotation.from_euler(
        "xyz",
        rotation_deg,
        degrees=True,
    )


def _rotation_to_euler(
    rotation: Rotation,
) -> list[float]:
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            UserWarning,
        )
        return rotation.as_euler(
            "xyz",
            degrees=True,
        ).tolist()


def _normalized_vector(
    values,
) -> np.ndarray:
    vector = np.asarray(
        values,
        dtype=float,
    )
    norm = np.linalg.norm(
        vector
    )

    if norm == 0:
        raise ValueError(
            "Direction vector cannot be zero."
        )

    return vector / norm


def _build_material(
    name: str,
    custom_n: float,
):
    if name == "Air":
        return AIR
    if name == "BK7":
        return BK7()
    if name == "FusedSilica":
        return FusedSilica()
    if name == "Custom":
        return ConstantIndex(
            custom_n,
            name=f"n={custom_n:g}",
        )

    raise ValueError(
        f"Unsupported material: {name}"
    )


def _material_refractive_index(
    name: str,
    custom_n: float,
) -> float:
    if name == "Custom":
        return float(custom_n)

    return float(
        _build_material(
            name,
            custom_n,
        ).n(632.8e-9)
    )


def _polyline_distances(
    polyline: np.ndarray,
    point: np.ndarray,
) -> list[float]:
    distances = []

    for start, end in zip(
        polyline[:-1],
        polyline[1:],
    ):
        distances.append(
            _point_segment_distance(
                point,
                start,
                end,
            )
        )

    return distances


def _point_segment_distance(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    segment = end - start
    length_sq = float(
        np.dot(segment, segment)
    )

    if length_sq == 0:
        return float(
            np.linalg.norm(
                point - start
            )
        )

    t = np.dot(
        point - start,
        segment,
    ) / length_sq
    t = np.clip(t, 0.0, 1.0)
    projection = start + t * segment
    return float(
        np.linalg.norm(
            point - projection
        )
    )


def _project_point(
    point,
    view: str,
) -> np.ndarray:
    i, j = VIEW_MAP[view]
    array = np.asarray(
        point,
        dtype=float,
    )
    return array[[i, j]]


def _project_vector(
    vector,
    view: str,
) -> np.ndarray:
    return _project_point(
        vector,
        view,
    )


def _screen_rotation_axis(
    view: str,
) -> int:
    axes = set(
        VIEW_MAP[view]
    )
    for axis in range(3):
        if axis not in axes:
            return axis
    raise ValueError(
        f"Invalid view: {view}"
    )


def _snap_drag_coordinate(
    value: float,
) -> float:
    return float(
        np.round(value)
    )




