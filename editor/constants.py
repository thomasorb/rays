from __future__ import annotations

from .registry import (
    REGISTRY,
    component_types,
)

# ==========================================================
# Materials
# ==========================================================

MATERIAL_NAMES = [
    "Air",
    "BK7",
    "FusedSilica",
    "Custom",
]

# ==========================================================
# Axes
# ==========================================================

AXIS_NAMES = [
    "x",
    "y",
    "z",
]

AXIS_INDEX = {
    "x": 0,
    "y": 1,
    "z": 2,
}

# ==========================================================
# Components
# ==========================================================

COMPONENT_TYPES = (
    component_types()
)

COMPONENT_COLORS = {

    name: definition.color

    for name, definition
    in REGISTRY.items()
}

THICK_COMPONENTS = {

    name

    for name, definition
    in REGISTRY.items()

    if "thickness"
    in definition.defaults
}
