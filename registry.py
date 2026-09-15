from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import importlib
import inspect
from pathlib import Path
import pkgutil
import re

from materials.material import Material


ROOT = Path(__file__).resolve().parent

MATERIAL_BASES = {
    "Material",
    "SellmeierMaterial",
}

OPTIC_BASES = {
    "OpticalElement",
    "PlanarElement",
    "OpticalInterface",
}


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    label: str
    kind: str
    required: bool
    default: object = None


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    label: str
    module: str
    factory: object
    parameters: tuple[ParameterSpec, ...]
    capabilities: tuple[str, ...]


def _humanize(
    name,
):
    if (
        name
        and name.upper() == name
    ):
        return name

    words = re.sub(
        r"(?<!^)(?=[A-Z])",
        " ",
        name.replace(
            "_",
            " ",
        ),
    )
    return " ".join(
        words.split()
    ).strip().title()


def _iter_modules(
    package_name,
):
    package_root = (
        ROOT / package_name
    )

    for module_info in pkgutil.iter_modules(
        [str(package_root)]
    ):
        if module_info.name.startswith(
            "_"
        ):
            continue

        yield importlib.import_module(
            f"{package_name}.{module_info.name}"
        )


def _parameter_kind(
    parameter,
):
    default = parameter.default

    if parameter.name in {
        "position",
        "direction",
        "optical_axis",
    }:
        return "vector"

    if parameter.name == "rotation":
        return "rotation"

    if default is inspect._empty:
        return "text"

    if isinstance(
        default,
        bool,
    ):
        return "bool"

    if isinstance(
        default,
        int,
    ):
        return "int"

    if isinstance(
        default,
        float,
    ):
        return "float"

    return "text"


def _parameter_specs(
    factory,
):
    if inspect.isclass(
        factory
    ):
        signature = inspect.signature(
            factory.__init__
        )
        parameters = list(
            signature.parameters.values()
        )[1:]
    else:
        signature = inspect.signature(
            factory
        )
        parameters = list(
            signature.parameters.values()
        )

    specs = []

    for parameter in parameters:
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue

        default = (
            None
            if parameter.default
            is inspect._empty
            else parameter.default
        )

        specs.append(
            ParameterSpec(
                name=parameter.name,
                label=_humanize(
                    parameter.name
                ),
                kind=_parameter_kind(
                    parameter
                ),
                required=(
                    parameter.default
                    is inspect._empty
                ),
                default=default,
            )
        )

    return tuple(specs)


def _capabilities(
    parameter_specs,
):
    names = {
        spec.name
        for spec in parameter_specs
    }

    capabilities = []

    if "position" in names:
        capabilities.append(
            "movable"
        )

    if names.intersection(
        {
            "rotation",
            "direction",
            "optical_axis",
        }
    ):
        capabilities.append(
            "rotatable"
        )

    return tuple(
        capabilities
    )


def _display_label(
    name,
    factory,
):
    display_name = getattr(
        factory,
        "DISPLAY_NAME",
        None,
    )

    if isinstance(
        display_name,
        str,
    ) and display_name:
        return display_name

    return _humanize(name)


def _material_specs_from_module(
    module,
):
    specs = []

    for name, member in inspect.getmembers(
        module
    ):
        if name.startswith("_"):
            continue

        if inspect.isclass(
            member
        ):
            if not issubclass(
                member,
                Material,
            ):
                continue

            if name in MATERIAL_BASES:
                continue

            parameter_specs = (
                _parameter_specs(
                    member
                )
            )

            specs.append(
                ComponentSpec(
                    name=name,
                    label=_display_label(
                        name,
                        member,
                    ),
                    module=module.__name__,
                    factory=member,
                    parameters=parameter_specs,
                    capabilities=(),
                )
            )
            continue

        if isinstance(
            member,
            Material,
        ):
            specs.append(
                ComponentSpec(
                    name=name,
                    label=getattr(
                        member,
                        "name",
                        _humanize(name),
                    ),
                    module=module.__name__,
                    factory=partial(
                        lambda value: value,
                        member,
                    ),
                    parameters=(),
                    capabilities=(),
                )
            )

    return specs


def discover_material_specs():
    discovered = {}

    for module in _iter_modules(
        "materials"
    ):
        for spec in (
            _material_specs_from_module(
                module
            )
        ):
            key = (
                spec.module,
                spec.name,
            )
            discovered[key] = spec

    return sorted(
        discovered.values(),
        key=lambda spec: (
            spec.label,
            spec.name,
        ),
    )


def discover_optic_specs():
    discovered = {}

    for module in _iter_modules(
        "optics"
    ):
        for name, member in inspect.getmembers(
            module,
            inspect.isclass,
        ):
            if name.startswith("_"):
                continue

            if member.__module__ != module.__name__:
                continue

            if name in OPTIC_BASES:
                continue

            parameter_specs = (
                _parameter_specs(
                    member
                )
            )

            discovered[
                (
                    module.__name__,
                    name,
                )
            ] = ComponentSpec(
                name=name,
                label=_humanize(
                    name
                ),
                module=module.__name__,
                factory=member,
                parameters=parameter_specs,
                capabilities=_capabilities(
                    parameter_specs
                ),
            )

    return sorted(
        discovered.values(),
        key=lambda spec: (
            spec.label,
            spec.name,
        ),
    )
