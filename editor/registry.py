from __future__ import annotations

import json
import importlib

from dataclasses import dataclass
from pathlib import Path


# ==========================================================
# Paths
# ==========================================================

COMPONENT_DEFINITIONS_DIR = (
    Path(__file__).parent
    / "component_definitions"
)


# ==========================================================
# Data model
# ==========================================================

@dataclass
class ComponentField:

    name: str

    label: str

    type: str = "float"


@dataclass
class ComponentDefinition:

    #
    # Component identity
    #

    type: str

    module: str

    class_name: str

    #
    # Editor metadata
    #

    prefix: str

    color: str

    #
    # Object creation
    #

    add_mode: str

    constructor_map: dict

    rotation_mode: str
    
    #
    # Defaults
    #

    defaults: dict

    #
    # Editor fields
    #

    fields: list[ComponentField]

    #
    # Debugging
    #

    file_name: str

    # ------------------------------------------------------

    def load_class(
        self,
    ):

        module = importlib.import_module(
            f"optics.{self.module}"
        )

        return getattr(
            module,
            self.class_name,
        )


# ==========================================================
# Loading
# ==========================================================

def load_component_definition(
    path: Path,
):
    data = json.loads(
        path.read_text()
    )

    fields = [

        ComponentField(**field)

        for field in data.get(
            "fields",
            []
        )
    ]

    return ComponentDefinition(

        type=data["type"],

        module=data["module"],

        class_name=data["class"],

        prefix=data["prefix"],

        color=data["color"],

        add_mode=data.get(
            "add_mode",
            "element",
        ),

        constructor_map=data.get(
            "constructor_map",
            {},
        ),

        defaults=data.get(
            "defaults",
            {},
        ),

        rotation_mode=data.get(
            "rotation_mode",
            "rotation",
        ),

        fields=fields,

        file_name=path.name,
    )


# ==========================================================
# Registry creation
# ==========================================================

def load_registry():

    registry = {}

    if not COMPONENT_DEFINITIONS_DIR.exists():

        return registry

    for file in sorted(
        COMPONENT_DEFINITIONS_DIR.glob(
            "*.json"
        )
    ):

        definition = (
            load_component_definition(
                file
            )
        )

        registry[
            definition.type
        ] = definition

    return registry


# ==========================================================
# Global registry
# ==========================================================

REGISTRY = load_registry()


# ==========================================================
# Convenience
# ==========================================================

def component_types():

    return sorted(
        REGISTRY.keys()
    )


def component_definition(
    component_type,
):
    return REGISTRY[
        component_type
    ]


def component_class(
    component_type,
):
    return component_definition(
        component_type
    ).load_class()


def component_defaults(
    component_type,
):
    return component_definition(
        component_type
    ).defaults


def component_fields(
    component_type,
):
    return component_definition(
        component_type
    ).fields


def component_color(
    component_type,
):
    return component_definition(
        component_type
    ).color
