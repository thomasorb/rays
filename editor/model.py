from pathlib import Path
import json
from dataclasses import asdict, dataclass, field

from optics.window import Window
from optics.detector import Detector
from optics.mirror import Mirror
from optics.source import Source
from optics.bundle_source import CircularBundleSource
from optics.compensator import Compensator
from optics.optical_interface import OpticalInterface
from optics.plate_beamsplitter import PlateBeamSplitter
from optics.circular_beamsplitter import CircularBeamSplitter
from optics.circular_plate_beamsplitter import CircularPlateBeamSplitter
from optics.corner_cube import CornerCube

from architectures.classic_michelson import ClassicMichelson
from system.optical_system import OpticalSystem

from materials.air import AIR
from materials.bk7 import BK7
from materials.constant_index import ConstantIndex
from materials.fused_silica import FusedSilica


from .specs import (
    SourceSpec,
    ComponentSpec,
)

from .utils import (
    _rotation_to_euler,
    _rotation_from_degrees,
    _normalized_vector,
    _build_material,
    _material_refractive_index,
)

from .registry import (
    component_definition,
    component_types,
)


class OpticalArchitectureModel:
    def __init__(
        self,
        components: list[ComponentSpec] | None = None,
        source: SourceSpec | None = None,
    ):
        self.components = (
            components
            if components is not None
            else self._default_components()
        )
        self.source = (
            source
            if source is not None
            else SourceSpec()
        )

    @classmethod
    def from_classic_michelson(
        cls,
    ) -> "OpticalArchitectureModel":
        architecture = ClassicMichelson(
            arm_length=50.0,
            compensator_distance=25.0,
            mirror_size=25.0,
            plate_size=25.0,
            plate_thickness=5.0,
            material=BK7(),
            detector_distance=50.0,
        )

        components = [
            ComponentSpec(
                component_type="PlateBeamSplitter",
                name="BS",
                position=architecture.bs.position.tolist(),
                rotation_deg=_rotation_to_euler(
                    architecture.bs.rotation
                ),
                width=float(
                    architecture.bs.front.width
                ),
                height=float(
                    architecture.bs.front.height
                ),
                thickness=float(
                    architecture.bs.thickness
                ),
                material="BK7",
                material_n=1.5,
                R=float(
                    architecture.bs.front.R
                ),
                T=float(
                    architecture.bs.front.T
                ),
                back_R=float(
                    architecture.bs.back.R
                ),
                back_T=float(
                    architecture.bs.back.T
                ),
            ),
            ComponentSpec(
                component_type="Compensator",
                name="COMP",
                position=architecture.compensator.position.tolist(),
                rotation_deg=_rotation_to_euler(
                    architecture.compensator.rotation
                ),
                width=float(
                    architecture.compensator.width
                ),
                height=float(
                    architecture.compensator.height
                ),
                thickness=float(
                    architecture.compensator.thickness
                ),
                material="BK7",
                material_n=1.5,
            ),
            ComponentSpec(
                component_type="Mirror",
                name="M1",
                position=architecture.mirror1.transform.position.tolist(),
                rotation_deg=_rotation_to_euler(
                    architecture.mirror1.transform.rotation
                ),
                width=float(
                    architecture.mirror1.width
                ),
                height=float(
                    architecture.mirror1.height
                ),
            ),
            ComponentSpec(
                component_type="Mirror",
                name="M2",
                position=architecture.mirror2.transform.position.tolist(),
                rotation_deg=_rotation_to_euler(
                    architecture.mirror2.transform.rotation
                ),
                width=float(
                    architecture.mirror2.width
                ),
                height=float(
                    architecture.mirror2.height
                ),
            ),
            ComponentSpec(
                component_type="Detector",
                name="DET1",
                position=architecture.detector1.transform.position.tolist(),
                rotation_deg=_rotation_to_euler(
                    architecture.detector1.transform.rotation
                ),
                width=float(
                    architecture.detector1.width
                ),
                height=float(
                    architecture.detector1.height
                ),
            ),
            ComponentSpec(
                component_type="Detector",
                name="DET2",
                position=architecture.detector2.transform.position.tolist(),
                rotation_deg=_rotation_to_euler(
                    architecture.detector2.transform.rotation
                ),
                width=float(
                    architecture.detector2.width
                ),
                height=float(
                    architecture.detector2.height
                ),
            ),
        ]

        return cls(
            components=components,
            source=SourceSpec(),
        )

    def _default_components(
        self,
    ) -> list[ComponentSpec]:
        return self.from_classic_michelson().components

    def _instantiate_component(
        self,
        spec,
        rotation,
    ):
        """
        Generic component instantiation from
        registry metadata.
        """

        definition = component_definition(
            spec.component_type
        )

        klass = definition.load_class()

        kwargs = {

            "name": spec.name,

            "position": spec.position,

        }

        if definition.rotation_mode == "optical_axis":

            kwargs[
                "optical_axis"
            ] = rotation.apply(
                [1, 0, 0]
            )

        else:

            kwargs[
                "rotation"
            ] = rotation

        #
        # Constructor mapping
        #

        for spec_name, ctor_name in (
            definition.constructor_map.items()
        ):

            value = getattr(
                spec,
                spec_name,
            )

            #
            # Material conversion
            #

            if spec_name == "material":

                value = _build_material(
                    spec.material,
                    spec.material_n,
                )

            kwargs[
                ctor_name
            ] = value

        return klass(
            **kwargs
        )
    
    def to_dict(
        self,
    ) -> dict:
        return {
            "source": asdict(self.source),
            "components": [
                component.to_dict()
                for component in self.components
            ],
        }

    def save_json(
        self,
        path: str | Path,
    ):
        target = Path(path)
        target.write_text(
            json.dumps(
                self.to_dict(),
                indent=2,
            )
        )

    @classmethod
    def load_json(
        cls,
        path: str | Path,
    ) -> "OpticalArchitectureModel":
        data = json.loads(
            Path(path).read_text()
        )

        source = SourceSpec(
            **data["source"]
        )

        components = [
            ComponentSpec.from_dict(item)
            for item in data["components"]
        ]

        return cls(
            components=components,
            source=source,
        )

    def unique_name(
        self,
        component_type: str,
    ) -> str:
        
        definition = component_definition(
            component_type
        )

        prefix = definition.prefix

        existing = {
            component.name
            for component in self.components
        }

        index = 1

        while True:

            name = f"{prefix}{index}"

            if name not in existing:

                return name

            index += 1

    
    def add_component(
        self,
        component_type: str,
    ) -> ComponentSpec:
        spec = ComponentSpec(
            component_type=component_type,
            name=self.unique_name(component_type),
            position=[0.0, 0.0, 0.0],
            rotation_deg=[0.0, 0.0, 0.0],
        )

        definition = component_definition(
            component_type
        )

        for key, value in (
            definition.defaults.items()
        ):

            if hasattr(
                spec,
                key,
            ):

                setattr(
                    spec,
                    key,
                    value,
                )

        self.components.append(spec)
        return spec

    def build_system(
        self,
    ) -> OpticalSystem:
        system = OpticalSystem()
        system.display_objects = {}
        system.set_source(
            self.build_source()
        )

        for spec in self.components:

            obj = self._add_component_to_system(
                system,
                spec,
            )

            if obj is not None:

                system.display_objects[
                    spec.name
                ] = obj

        return system

    def build_source(
        self,
    ):
        direction = _normalized_vector(
            self.source.direction
        )

        if self.source.kind == "single":
            return Source(
                position=self.source.position,
                direction=direction,
                wavelength=float(
                    self.source.wavelength
                ),
            )

        return CircularBundleSource(
            position=self.source.position,
            direction=direction,
            wavelength=float(
                self.source.wavelength
            ),
            radius=float(
                self.source.radius
            ),
            n_rays=int(
                self.source.n_rays
            ),
        )

    def _add_component_to_system(
        self,
        system,
        spec: ComponentSpec,
    ):

        rotation = _rotation_from_degrees(
            spec.rotation_deg
        )

        definition = component_definition(
            spec.component_type
        )

        obj = self._instantiate_component(
            spec,
            rotation,
        )

        #
        # Detector
        #

        if definition.add_mode == "detector":

            system.add_detector(
                obj
            )

            return obj

        #
        # Assemblies
        #

        if definition.add_mode == "assembly":

            obj.add_to_system(
                system
            )

            return obj

        #
        # Standard optical element
        #

        system.add_element(
            obj
        )

        return obj

    def _build_thick_component(
        self,
        spec: ComponentSpec,
        rotation: Rotation,
        component_class,
    ):
        kwargs = dict(
            name=spec.name,
            position=spec.position,
            rotation=rotation,
            thickness=spec.thickness,
            width=spec.width,
            height=spec.height,
            material=_build_material(
                spec.material,
                spec.material_n,
            ),
        )

        if component_class is PlateBeamSplitter:
            kwargs.update(
                R=spec.R,
                T=spec.T,
                back_R=spec.back_R,
                back_T=spec.back_T,
            )

        return component_class(
            **kwargs
        )
