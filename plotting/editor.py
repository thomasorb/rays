from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
from matplotlib.widgets import TextBox

from plotting.views import plot_element
from plotting.views import plot_ray
from registry import discover_material_specs
from registry import discover_optic_specs
from matplotlib.figure import Figure
from matplotlib.path import Path as MplPath
from scipy.spatial.transform import Rotation

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg,
    )
    TK_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    tk = None
    ttk = None
    filedialog = None
    messagebox = None
    FigureCanvasTkAgg = None
    TK_IMPORT_ERROR = exc

from architectures.classic_michelson import ClassicMichelson
from materials.air import AIR
from materials.bk7 import BK7
from materials.constant_index import ConstantIndex
from materials.fused_silica import FusedSilica
from optics.bundle_source import CircularBundleSource
from optics.compensator import Compensator
from optics.detector import Detector
from optics.mirror import Mirror
from optics.optical_interface import OpticalInterface
from optics.plate_beamsplitter import PlateBeamSplitter
from optics.source import Source
from optics.window import Window
from plotting.views import VIEW_MAP
from system.optical_system import OpticalSystem


THICK_COMPONENTS = {
    "Window",
    "Compensator",
    "PlateBeamSplitter",
}

COMPONENT_TYPES = [
    "Mirror",
    "Detector",
    "OpticalInterface",
    "Window",
    "Compensator",
    "PlateBeamSplitter",
]

MATERIAL_NAMES = [
    "Air",
    "BK7",
    "FusedSilica",
    "Custom",
]

AXIS_NAMES = ["x", "y", "z"]
AXIS_INDEX = {
    "x": 0,
    "y": 1,
    "z": 2,
}

COMPONENT_COLORS = {
    "Mirror": "royalblue",
    "Detector": "forestgreen",
    "OpticalInterface": "dimgray",
    "Window": "purple",
    "Compensator": "darkorange",
    "PlateBeamSplitter": "crimson",
}


@dataclass
class SourceSpec:
    kind: str = "bundle"
    position: list[float] = field(
        default_factory=lambda: [-50.0, 0.0, 0.0]
    )
    direction: list[float] = field(
        default_factory=lambda: [1.0, 0.0, 0.0]
    )
    wavelength: float = 632.8e-9
    radius: float = 2.0
    n_rays: int = 31


@dataclass
class ComponentSpec:
    component_type: str
    name: str
    position: list[float]
    rotation_deg: list[float]
    width: float = 25.0
    height: float = 25.0
    thickness: float = 5.0
    material: str = "BK7"
    material_n: float = 1.5
    material1: str = "Air"
    material1_n: float = 1.0
    material2: str = "Air"
    material2_n: float = 1.5
    R: float = 0.0
    T: float = 1.0
    back_R: float = 0.0
    back_T: float = 1.0
    phase_reflection: float = float(np.pi / 2)

    def to_dict(
        self,
    ) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict,
    ) -> "ComponentSpec":
        return cls(**data)


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
        prefix_map = {
            "Mirror": "M",
            "Detector": "DET",
            "OpticalInterface": "IF",
            "Window": "WIN",
            "Compensator": "COMP",
            "PlateBeamSplitter": "BS",
        }

        prefix = prefix_map[component_type]
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

        if component_type == "Detector":
            spec.rotation_deg = [0.0, -90.0, 0.0]
        elif component_type == "PlateBeamSplitter":
            spec.rotation_deg = [0.0, 135.0, 0.0]
            spec.R = 0.5
            spec.T = 0.5
            spec.back_T = 1.0
        elif component_type == "Compensator":
            spec.rotation_deg = [0.0, 135.0, 0.0]
        elif component_type == "OpticalInterface":
            spec.material1 = "Air"
            spec.material2 = "BK7"
            spec.material2_n = 1.5
            spec.R = 0.5
            spec.T = 0.5

        self.components.append(spec)
        return spec

    def build_system(
        self,
    ) -> OpticalSystem:
        system = OpticalSystem()
        system.set_source(
            self.build_source()
        )

        for spec in self.components:
            self._add_component_to_system(
                system,
                spec,
            )

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
        system: OpticalSystem,
        spec: ComponentSpec,
    ):
        rotation = _rotation_from_degrees(
            spec.rotation_deg
        )

        if spec.component_type == "Mirror":
            system.add_element(
                Mirror(
                    name=spec.name,
                    position=spec.position,
                    rotation=rotation,
                    width=spec.width,
                    height=spec.height,
                )
            )
            return

        if spec.component_type == "Detector":
            system.add_detector(
                Detector(
                    name=spec.name,
                    position=spec.position,
                    rotation=rotation,
                    width=spec.width,
                    height=spec.height,
                )
            )
            return

        if spec.component_type == "OpticalInterface":
            system.add_element(
                OpticalInterface(
                    name=spec.name,
                    position=spec.position,
                    rotation=rotation,
                    width=spec.width,
                    height=spec.height,
                    material1=_build_material(
                        spec.material1,
                        spec.material1_n,
                    ),
                    material2=_build_material(
                        spec.material2,
                        spec.material2_n,
                    ),
                    R=spec.R,
                    T=spec.T,
                    phase_reflection=spec.phase_reflection,
                )
            )
            return

        if spec.component_type == "Window":
            self._build_thick_component(
                spec,
                rotation,
                Window,
            ).add_to_system(system)
            return

        if spec.component_type == "Compensator":
            self._build_thick_component(
                spec,
                rotation,
                Compensator,
            ).add_to_system(system)
            return

        if spec.component_type == "PlateBeamSplitter":
            self._build_thick_component(
                spec,
                rotation,
                PlateBeamSplitter,
            ).add_to_system(system)
            return

        raise ValueError(
            f"Unsupported component type: {spec.component_type}"
        )

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


class OpticalEditorApp:
    def __init__(
        self,
        root: tk.Tk,
        model: OpticalArchitectureModel | None = None,
    ):
        if TK_IMPORT_ERROR is not None:
            raise RuntimeError(
                "tkinter is required to use the optical editor."
            ) from TK_IMPORT_ERROR

        self.root = root
        self.model = (
            model
            if model is not None
            else OpticalArchitectureModel()
        )

        self.current_view = "xz"
        self.current_rays = None
        self.selected_index: int | None = None
        self.panel_mode = "component"
        self.component_vars: dict[str, tk.Variable] = {}
        self.trace_vars: dict[str, tk.Variable] = {}
        self.move_vars: dict[str, tk.Variable] = {}
        self.drag_state: dict | None = None
        self.last_limits: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}
        self.add_type_var = tk.StringVar(
            value="Mirror"
        )
        self.move_component_var = tk.StringVar()
        self.move_axis_var = tk.StringVar(
            value="x"
        )
        self.move_detector_1_var = tk.StringVar()
        self.move_detector_2_var = tk.StringVar()

        self.root.title(
            "Optical System Editor"
        )
        self.root.geometry(
            "1400x850"
        )

        self._build_ui()
        self._rebuild_system()
        self._show_component_panel()
        self._render_scene(
            preserve_limits=False
        )

    def _build_ui(
        self,
    ):
        self.root.rowconfigure(
            1,
            weight=1,
        )
        self.root.columnconfigure(
            0,
            weight=1,
        )

        self.toolbar = ttk.Frame(
            self.root,
            padding=6,
        )
        self.toolbar.grid(
            row=0,
            column=0,
            sticky="ew",
        )

        for index, label in enumerate(
            ["MOVE", "TRACE", "QUIT", "XY", "XZ", "YZ"]
        ):
            command = {
                "MOVE": self._show_move_panel,
                "TRACE": self._show_trace_panel,
                "QUIT": self.root.destroy,
                "XY": lambda: self._set_view("xy"),
                "XZ": lambda: self._set_view("xz"),
                "YZ": lambda: self._set_view("yz"),
            }[label]

            ttk.Button(
                self.toolbar,
                text=label,
                command=command,
            ).grid(
                row=0,
                column=index,
                padx=4,
            )

        self.notebook = ttk.Notebook(
            self.root
        )
        self.notebook.grid(
            row=1,
            column=0,
            sticky="nsew",
        )

        self.main_tab = ttk.Frame(
            self.notebook
        )
        self.debug_tab = ttk.Frame(
            self.notebook
        )
        self.notebook.add(
            self.main_tab,
            text="Main",
        )
        self.notebook.add(
            self.debug_tab,
            text="Debug",
        )

        self.main_tab.rowconfigure(
            0,
            weight=1,
        )
        self.main_tab.columnconfigure(
            0,
            weight=3,
        )
        self.main_tab.columnconfigure(
            1,
            weight=1,
        )

        self.geometry_frame = ttk.Frame(
            self.main_tab,
            padding=(6, 6, 3, 6),
        )
        self.geometry_frame.grid(
            row=0,
            column=0,
            sticky="nsew",
        )
        self.geometry_frame.rowconfigure(
            0,
            weight=1,
        )
        self.geometry_frame.columnconfigure(
            0,
            weight=1,
        )

        self.panel_frame = ttk.Frame(
            self.main_tab,
            padding=(3, 6, 6, 6),
        )
        self.panel_frame.grid(
            row=0,
            column=1,
            sticky="nsew",
        )
        self.panel_frame.rowconfigure(
            2,
            weight=1,
        )
        self.panel_frame.columnconfigure(
            0,
            weight=1,
        )

        self.figure = Figure(
            figsize=(10, 7),
            dpi=100,
        )
        self.axes = self.figure.add_subplot(
            111
        )
        self.canvas = FigureCanvasTkAgg(
            self.figure,
            master=self.geometry_frame,
        )
        self.canvas.get_tk_widget().grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        self.figure.canvas.mpl_connect(
            "button_press_event",
            self._on_button_press,
        )
        self.figure.canvas.mpl_connect(
            "motion_notify_event",
            self._on_mouse_move,
        )
        self.figure.canvas.mpl_connect(
            "button_release_event",
            self._on_button_release,
        )
        self.figure.canvas.mpl_connect(
            "scroll_event",
            self._on_scroll,
        )

        static_controls = ttk.Frame(
            self.panel_frame
        )
        static_controls.grid(
            row=0,
            column=0,
            sticky="ew",
        )
        static_controls.columnconfigure(
            1,
            weight=1,
        )

        ttk.Button(
            static_controls,
            text="Save",
            command=self._save_architecture,
        ).grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 4),
            pady=(0, 4),
        )
        ttk.Button(
            static_controls,
            text="Load",
            command=self._load_architecture,
        ).grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(4, 0),
            pady=(0, 4),
        )

        add_row = ttk.Frame(
            static_controls
        )
        add_row.grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="ew",
        )
        add_row.columnconfigure(
            0,
            weight=1,
        )

        ttk.Combobox(
            add_row,
            textvariable=self.add_type_var,
            values=COMPONENT_TYPES,
            state="readonly",
        ).grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 4),
            pady=(0, 4),
        )

        ttk.Button(
            add_row,
            text="Add",
            command=self._add_component,
        ).grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(4, 0),
            pady=(0, 4),
        )

        self.remove_button = ttk.Button(
            static_controls,
            text="Remove selected",
            command=self._remove_selected_component,
        )
        self.remove_button.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="ew",
        )

        self.panel_title = ttk.Label(
            self.panel_frame,
            text="Component",
            font=("", 11, "bold"),
        )
        self.panel_title.grid(
            row=1,
            column=0,
            sticky="w",
            pady=(10, 6),
        )

        self.panel_body = ttk.Frame(
            self.panel_frame
        )
        self.panel_body.grid(
            row=2,
            column=0,
            sticky="nsew",
        )
        self.panel_body.columnconfigure(
            1,
            weight=1,
        )

        self.debug_text = tk.Text(
            self.debug_tab,
            wrap="word",
            state="disabled",
        )
        self.debug_text.pack(
            fill="both",
            expand=True,
        )

    def _log(
        self,
        message: str,
    ):
        self.debug_text.configure(
            state="normal"
        )
        self.debug_text.insert(
            "end",
            message.rstrip() + "\n",
        )
        self.debug_text.see("end")
        self.debug_text.configure(
            state="disabled"
        )

    def _run_logged(
        self,
        action,
        success_message: str | None = None,
    ):
        buffer = io.StringIO()

        try:
            with redirect_stdout(buffer), redirect_stderr(buffer):
                result = action()
        except Exception as exc:
            output = buffer.getvalue().strip()
            if output:
                self._log(output)
            self._log(
                f"ERROR: {exc}"
            )
            messagebox.showerror(
                "Error",
                str(exc),
            )
            return None

        output = buffer.getvalue().strip()
        if output:
            self._log(output)

        if success_message:
            self._log(success_message)

        return result

    def _clear_panel(
        self,
    ):
        for child in self.panel_body.winfo_children():
            child.destroy()

    def _show_component_panel(
        self,
    ):
        self.panel_mode = "component"
        self.panel_title.configure(
            text="Component parameters"
        )
        self._clear_panel()
        self.component_vars = {}

        if self.selected_index is None:
            ttk.Label(
                self.panel_body,
                text="Select a component on the plot.",
            ).grid(
                row=0,
                column=0,
                sticky="w",
            )
            self.remove_button.state(
                ["disabled"]
            )
            return

        self.remove_button.state(
            ["!disabled"]
        )
        spec = self.model.components[
            self.selected_index
        ]

        row = 0

        self.component_vars["name"] = tk.StringVar(
            value=spec.name
        )
        self._add_entry(
            "Name",
            self.component_vars["name"],
            row,
        )
        row += 1

        ttk.Label(
            self.panel_body,
            text="Position [mm]",
        ).grid(
            row=row,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(6, 0),
        )
        row += 1

        for axis, value in zip(
            AXIS_NAMES,
            spec.position,
        ):
            variable = tk.DoubleVar(
                value=float(value)
            )
            self.component_vars[
                f"position_{axis}"
            ] = variable
            self._add_entry(
                axis.upper(),
                variable,
                row,
            )
            row += 1

        ttk.Label(
            self.panel_body,
            text="Rotation [deg]",
        ).grid(
            row=row,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(6, 0),
        )
        row += 1

        for axis, value in zip(
            AXIS_NAMES,
            spec.rotation_deg,
        ):
            variable = tk.DoubleVar(
                value=float(value)
            )
            self.component_vars[
                f"rotation_{axis}"
            ] = variable
            self._add_entry(
                axis.upper(),
                variable,
                row,
            )
            row += 1

        for key, label, value in self._component_scalar_fields(
            spec
        ):
            variable = tk.DoubleVar(
                value=float(value)
            )
            self.component_vars[key] = variable
            self._add_entry(
                label,
                variable,
                row,
            )
            row += 1

        for key, label, value in self._component_material_fields(
            spec
        ):
            variable = tk.StringVar(
                value=value
            )
            self.component_vars[key] = variable
            self._add_combobox(
                label,
                variable,
                MATERIAL_NAMES,
                row,
            )
            row += 1

            numeric_key = (
                f"{key}_n"
            )
            if hasattr(spec, numeric_key):
                numeric_var = tk.DoubleVar(
                    value=float(
                        _material_refractive_index(
                            value,
                            getattr(
                                spec,
                                numeric_key,
                                1.5,
                            ),
                        )
                    )
                )
                self.component_vars[
                    numeric_key
                ] = numeric_var
                self._add_entry(
                    f"{label} n",
                    numeric_var,
                    row,
                )
                row += 1

        ttk.Button(
            self.panel_body,
            text="Apply",
            command=self._apply_component_changes,
        ).grid(
            row=row,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(10, 0),
        )

    def _refresh_move_ranges(
        self,
    ):
        if self.panel_mode != "move":
            return

        spec = self._find_component_by_name(
            self.move_component_var.get()
        )
        if spec is None:
            return

        axis_index = AXIS_INDEX[
            self.move_axis_var.get()
        ]
        center_value = float(
            spec.position[axis_index]
        )

        if "minimum" in self.move_vars:
            self.move_vars["minimum"].set(
                center_value - 1.0
            )
        if "maximum" in self.move_vars:
            self.move_vars["maximum"].set(
                center_value + 1.0
            )

    def _show_trace_panel(
        self,
    ):
        self.panel_mode = "trace"
        self.panel_title.configure(
            text="Ray tracing parameters"
        )
        self._clear_panel()
        self.trace_vars = {}
        self.remove_button.state(
            ["disabled"]
        )

        source = self.model.source
        row = 0

        self.trace_vars["kind"] = tk.StringVar(
            value=source.kind
        )
        self._add_combobox(
            "Source type",
            self.trace_vars["kind"],
            ["single", "bundle"],
            row,
        )
        row += 1

        ttk.Label(
            self.panel_body,
            text="Position [mm]",
        ).grid(
            row=row,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(6, 0),
        )
        row += 1

        for axis, value in zip(
            AXIS_NAMES,
            source.position,
        ):
            variable = tk.DoubleVar(
                value=float(value)
            )
            self.trace_vars[
                f"position_{axis}"
            ] = variable
            self._add_entry(
                axis.upper(),
                variable,
                row,
            )
            row += 1

        ttk.Label(
            self.panel_body,
            text="Direction",
        ).grid(
            row=row,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(6, 0),
        )
        row += 1

        for axis, value in zip(
            AXIS_NAMES,
            source.direction,
        ):
            variable = tk.DoubleVar(
                value=float(value)
            )
            self.trace_vars[
                f"direction_{axis}"
            ] = variable
            self._add_entry(
                axis.upper(),
                variable,
                row,
            )
            row += 1

        for key, label, value in [
            (
                "wavelength",
                "Wavelength [m]",
                source.wavelength,
            ),
            (
                "radius",
                "Bundle radius [mm]",
                source.radius,
            ),
            (
                "n_rays",
                "Number of rays",
                source.n_rays,
            ),
        ]:
            variable = (
                tk.IntVar(value=int(value))
                if key == "n_rays"
                else tk.DoubleVar(value=float(value))
            )
            self.trace_vars[key] = variable
            self._add_entry(
                label,
                variable,
                row,
            )
            row += 1

        ttk.Button(
            self.panel_body,
            text="Apply",
            command=self._apply_trace_settings,
        ).grid(
            row=row,
            column=0,
            sticky="ew",
            pady=(10, 0),
            padx=(0, 4),
        )
        ttk.Button(
            self.panel_body,
            text="Trace",
            command=self._trace_rays,
        ).grid(
            row=row,
            column=1,
            sticky="ew",
            pady=(10, 0),
            padx=(4, 0),
        )

    def _show_move_panel(
        self,
    ):
        self.panel_mode = "move"
        self.panel_title.configure(
            text="Movement parameters"
        )
        self._clear_panel()
        self.move_vars = {}
        self.remove_button.state(
            ["disabled"]
        )

        names = [
            component.name
            for component in self.model.components
        ]

        if names and (
            self.move_component_var.get()
            not in names
        ):
            selected = (
                self.model.components[
                    self.selected_index
                ].name
                if self.selected_index is not None
                else names[0]
            )
            self.move_component_var.set(
                selected
            )

        row = 0
        self._add_combobox(
            "Component",
            self.move_component_var,
            names,
            row,
            callback=self._refresh_move_ranges,
        )
        row += 1

        self._add_combobox(
            "Axis",
            self.move_axis_var,
            AXIS_NAMES,
            row,
            callback=self._refresh_move_ranges,
        )
        row += 1

        detector_names = [
            component.name
            for component in self.model.components
            if component.component_type == "Detector"
        ]

        if detector_names:
            if (
                self.move_detector_1_var.get()
                not in detector_names
            ):
                self.move_detector_1_var.set(
                    detector_names[0]
                )
            if (
                self.move_detector_2_var.get()
                not in detector_names
            ):
                default_name = (
                    detector_names[1]
                    if len(detector_names) > 1
                    else detector_names[0]
                )
                self.move_detector_2_var.set(
                    default_name
                )

        self._add_combobox(
            "Detector 1",
            self.move_detector_1_var,
            detector_names,
            row,
        )
        row += 1

        self._add_combobox(
            "Detector 2",
            self.move_detector_2_var,
            detector_names,
            row,
        )
        row += 1

        selected_spec = self._find_component_by_name(
            self.move_component_var.get()
        )
        axis_index = AXIS_INDEX[
            self.move_axis_var.get()
        ]
        center_value = (
            selected_spec.position[
                axis_index
            ]
            if selected_spec is not None
            else 0.0
        )

        for key, label, value in [
            (
                "minimum",
                "Min position [mm]",
                center_value - 1.0,
            ),
            (
                "maximum",
                "Max position [mm]",
                center_value + 1.0,
            ),
            (
                "points",
                "Number of points",
                21,
            ),
        ]:
            variable = (
                tk.IntVar(value=int(value))
                if key == "points"
                else tk.DoubleVar(value=float(value))
            )
            self.move_vars[key] = variable
            self._add_entry(
                label,
                variable,
                row,
            )
            row += 1

        ttk.Button(
            self.panel_body,
            text="Run move scan",
            command=self._run_move_scan,
        ).grid(
            row=row,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(10, 0),
        )

    def _component_scalar_fields(
        self,
        spec: ComponentSpec,
    ) -> list[tuple[str, str, float]]:
        fields = [
            (
                "width",
                "Width [mm]",
                spec.width,
            ),
            (
                "height",
                "Height [mm]",
                spec.height,
            ),
        ]

        if spec.component_type in THICK_COMPONENTS:
            fields.append(
                (
                    "thickness",
                    "Thickness [mm]",
                    spec.thickness,
                )
            )

        if spec.component_type == "OpticalInterface":
            fields.extend(
                [
                    ("R", "Reflectance", spec.R),
                    ("T", "Transmittance", spec.T),
                    (
                        "phase_reflection",
                        "Reflection phase [rad]",
                        spec.phase_reflection,
                    ),
                ]
            )

        if spec.component_type == "PlateBeamSplitter":
            fields.extend(
                [
                    ("R", "Front reflectance", spec.R),
                    ("T", "Front transmittance", spec.T),
                    ("back_R", "Back reflectance", spec.back_R),
                    ("back_T", "Back transmittance", spec.back_T),
                ]
            )

        return fields

    def _component_material_fields(
        self,
        spec: ComponentSpec,
    ) -> list[tuple[str, str, str]]:
        if spec.component_type in THICK_COMPONENTS:
            return [
                (
                    "material",
                    "Material",
                    spec.material,
                )
            ]

        if spec.component_type == "OpticalInterface":
            return [
                (
                    "material1",
                    "Material 1",
                    spec.material1,
                ),
                (
                    "material2",
                    "Material 2",
                    spec.material2,
                ),
            ]

        return []

    def _add_entry(
        self,
        label: str,
        variable: tk.Variable,
        row: int,
    ):
        ttk.Label(
            self.panel_body,
            text=label,
        ).grid(
            row=row,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=2,
        )
        entry = ttk.Entry(
            self.panel_body,
            textvariable=variable,
        )
        entry.grid(
            row=row,
            column=1,
            sticky="ew",
            pady=2,
        )
        entry.bind(
            "<Return>",
            lambda _event: self._apply_active_panel(),
        )

    def _add_combobox(
        self,
        label: str,
        variable: tk.Variable,
        values: list[str],
        row: int,
        callback=None,
    ):
        ttk.Label(
            self.panel_body,
            text=label,
        ).grid(
            row=row,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=2,
        )
        combobox = ttk.Combobox(
            self.panel_body,
            textvariable=variable,
            values=values,
            state="readonly",
        )
        combobox.grid(
            row=row,
            column=1,
            sticky="ew",
            pady=2,
        )
        combobox.bind(
            "<<ComboboxSelected>>",
            lambda _event: (
                callback()
                if callback is not None
                else self._apply_active_panel()
            ),
        )

    def _apply_active_panel(
        self,
    ):
        if self.panel_mode == "component":
            self._apply_component_changes()
        elif self.panel_mode == "trace":
            self._apply_trace_settings()

    def _apply_component_changes(
        self,
    ):
        if self.selected_index is None:
            return

        spec = self.model.components[
            self.selected_index
        ]
        old_state = spec.to_dict()

        try:
            spec.name = self.component_vars[
                "name"
            ].get().strip()
            if not spec.name:
                raise ValueError(
                    "Component name cannot be empty."
                )

            spec.position = [
                float(
                    self.component_vars[
                        f"position_{axis}"
                    ].get()
                )
                for axis in AXIS_NAMES
            ]

            spec.rotation_deg = [
                float(
                    self.component_vars[
                        f"rotation_{axis}"
                    ].get()
                )
                for axis in AXIS_NAMES
            ]

            for key, _, _ in self._component_scalar_fields(
                spec
            ):
                setattr(
                    spec,
                    key,
                    float(
                        self.component_vars[key].get()
                    ),
                )

            for key, _, _ in self._component_material_fields(
                spec
            ):
                material_name = self.component_vars[
                    key
                ].get()
                setattr(
                    spec,
                    key,
                    material_name,
                )
                numeric_key = (
                    f"{key}_n"
                )
                if numeric_key in self.component_vars:
                    numeric_value = float(
                        self.component_vars[
                            numeric_key
                        ].get()
                    )
                    numeric_value = _material_refractive_index(
                        material_name,
                        numeric_value,
                    )
                    setattr(
                        spec,
                        numeric_key,
                        numeric_value,
                    )

            self._validate_component(
                spec
            )
            self._clear_rays()
            self._rebuild_system()

        except Exception as exc:
            self.model.components[
                self.selected_index
            ] = ComponentSpec.from_dict(
                old_state
            )
            self._rebuild_system()
            self._show_component_panel()
            messagebox.showerror(
                "Invalid component parameters",
                f"Invalid component parameters: {exc}",
            )
            return
        self._render_scene()
        self._show_component_panel()

    def _apply_trace_settings(
        self,
    ) -> bool:
        old_state = asdict(
            self.model.source
        )

        try:
            self.model.source.kind = (
                self.trace_vars["kind"].get()
            )
            self.model.source.position = [
                float(
                    self.trace_vars[
                        f"position_{axis}"
                    ].get()
                )
                for axis in AXIS_NAMES
            ]
            self.model.source.direction = [
                float(
                    self.trace_vars[
                        f"direction_{axis}"
                    ].get()
                )
                for axis in AXIS_NAMES
            ]
            self.model.source.wavelength = float(
                self.trace_vars[
                    "wavelength"
                ].get()
            )
            self.model.source.radius = float(
                self.trace_vars[
                    "radius"
                ].get()
            )
            self.model.source.n_rays = int(
                self.trace_vars[
                    "n_rays"
                ].get()
            )
            self._clear_rays()
            self._rebuild_system()

        except Exception as exc:
            self.model.source = SourceSpec(
                **old_state
            )
            self._rebuild_system()
            self._show_trace_panel()
            messagebox.showerror(
                "Invalid trace settings",
                f"Invalid trace settings: {exc}",
            )
            return False

        self._render_scene()
        return True

    def _validate_component(
        self,
        spec: ComponentSpec,
    ):
        names = [
            component.name
            for component in self.model.components
        ]
        if names.count(spec.name) > 1:
            raise ValueError(
                "Component names must be unique."
            )

        if spec.width <= 0 or spec.height <= 0:
            raise ValueError(
                "Width and height must be positive."
            )

        if spec.component_type in THICK_COMPONENTS and spec.thickness <= 0:
            raise ValueError(
                "Thickness must be positive."
            )

        if spec.component_type == "OpticalInterface":
            if (
                spec.R < 0
                or spec.R > 1
                or spec.T < 0
                or spec.T > 1
            ):
                raise ValueError(
                    "R and T must be between 0 and 1."
                )
            if spec.R + spec.T > 1:
                raise ValueError(
                    "Optical interface requires R + T <= 1."
                )

        if spec.component_type == "PlateBeamSplitter":
            for key, value in {
                "front_R": spec.R,
                "front_T": spec.T,
                "back_R": spec.back_R,
                "back_T": spec.back_T,
            }.items():
                if value < 0 or value > 1:
                    raise ValueError(
                        f"{key} must be between 0 and 1."
                    )
            if spec.R + spec.T > 1:
                raise ValueError(
                    "Front surface requires R + T <= 1."
                )
            if spec.back_R + spec.back_T > 1:
                raise ValueError(
                    "Back surface requires R + T <= 1."
                )

    def _rebuild_system(
        self,
    ):
        self.system = self.model.build_system()

    def _trace_rays(
        self,
    ):
        if not self._apply_trace_settings():
            return

        def action():
            self.system = self.model.build_system()
            rays = self.system.trace()
            return rays

        rays = self._run_logged(
            action,
            success_message="Ray tracing completed.",
        )

        if rays is None:
            return

        self.current_rays = rays
        self._render_scene()

        if self.system.detectors:
            for detector in self.system.detectors:
                self._log(
                    f"{detector.name}: intensity={detector.intensity:.6f}, power={detector.power:.6f}"
                )

    def _run_move_scan(
        self,
    ):
        component = self._find_component_by_name(
            self.move_component_var.get()
        )

        if component is None:
            messagebox.showerror(
                "Error",
                "Select a component to move.",
            )
            return

        axis_name = self.move_axis_var.get()
        axis_index = AXIS_INDEX[
            axis_name
        ]
        minimum = float(
            self.move_vars[
                "minimum"
            ].get()
        )
        maximum = float(
            self.move_vars[
                "maximum"
            ].get()
        )
        points = int(
            self.move_vars[
                "points"
            ].get()
        )

        if points < 2:
            messagebox.showerror(
                "Error",
                "Number of points must be at least 2."
            )
            return

        if minimum > maximum:
            messagebox.showerror(
                "Error",
                "Min position must be less than or equal to max position.",
            )
            return

        detector_1_name = self.move_detector_1_var.get()
        detector_2_name = self.move_detector_2_var.get()

        if not detector_1_name or not detector_2_name:
            messagebox.showerror(
                "Error",
                "Select two detectors for the move scan."
            )
            return

        positions = np.linspace(
            minimum,
            maximum,
            points,
        )
        available_detectors = {
            detector.name
            for detector in self.system.detectors
        }
        missing_detectors = [
            name
            for name in [
                detector_1_name,
                detector_2_name,
            ]
            if name not in available_detectors
        ]

        if missing_detectors:
            messagebox.showerror(
                "Error",
                "Selected detectors are not available in the current architecture.",
            )
            return

        original_position = list(
            component.position
        )
        had_rays = self.current_rays is not None

        def action():
            detector_1_values = []
            detector_2_values = []

            try:
                for value in positions:
                    component.position[axis_index] = float(
                        value
                    )
                    system = self.model.build_system()
                    system.trace()
                    detector_1 = system.detector_by_name(
                        detector_1_name
                    )
                    detector_2 = system.detector_by_name(
                        detector_2_name
                    )

                    detector_1_values.append(
                        detector_1.intensity
                    )
                    detector_2_values.append(
                        detector_2.intensity
                    )
            finally:
                component.position = list(
                    original_position
                )

            return (
                np.asarray(detector_1_values),
                np.asarray(detector_2_values),
            )

        result = None
        try:
            result = self._run_logged(
                action,
                success_message="Move scan completed.",
            )
        finally:
            component.position = list(
                original_position
            )
            self._rebuild_system()
            if had_rays:
                self.current_rays = self.system.trace()
            else:
                self._clear_rays()
            self._render_scene()

        if result is None:
            return

        detector_1, detector_2 = result
        self._show_move_result_window(
            component.name,
            axis_name,
            positions,
            detector_1,
            detector_2,
            detector_1_name,
            detector_2_name,
        )

    def _show_move_result_window(
        self,
        component_name: str,
        axis_name: str,
        positions: np.ndarray,
        detector_1: np.ndarray,
        detector_2: np.ndarray,
        detector_1_name: str,
        detector_2_name: str,
    ):
        window = tk.Toplevel(
            self.root
        )
        window.title(
            f"Move scan - {component_name}"
        )
        window.geometry(
            "900x500"
        )

        figure = Figure(
            figsize=(8, 4),
            dpi=100,
        )
        ax = figure.add_subplot(
            111
        )
        ax.plot(
            positions,
            detector_1,
            label=detector_1_name,
        )
        ax.plot(
            positions,
            detector_2,
            label=detector_2_name,
        )
        ax.set_xlabel(
            f"{component_name} {axis_name.upper()} position [mm]"
        )
        ax.set_ylabel(
            "Intensity"
        )
        ax.grid(True)
        ax.legend()

        canvas = FigureCanvasTkAgg(
            figure,
            master=window,
        )
        canvas.get_tk_widget().pack(
            fill="both",
            expand=True,
        )
        canvas.draw_idle()

        def on_close():
            figure.clear()
            window.destroy()

        window.protocol(
            "WM_DELETE_WINDOW",
            on_close,
        )

    def _set_view(
        self,
        view: str,
    ):
        self.current_view = view
        self._render_scene(
            preserve_limits=False
        )

    def _clear_rays(
        self,
    ):
        self.current_rays = None

    def _add_component(
        self,
    ):
        spec = self.model.add_component(
            self.add_type_var.get()
        )
        new_index = len(self.model.components) - 1

        try:
            self.selected_index = new_index
            self._clear_rays()
            self._rebuild_system()
        except Exception as exc:
            self.model.components.pop()
            self.selected_index = None
            self._rebuild_system()
            messagebox.showerror(
                "Invalid component parameters",
                f"Invalid component parameters: {exc}",
            )
            return

        self._show_component_panel()
        self._render_scene()
        self._log(
            f"Added component {spec.name}."
        )

    def _remove_selected_component(
        self,
    ):
        if self.selected_index is None:
            return

        removed_index = self.selected_index
        removed = self.model.components.pop(
            self.selected_index
        )
        self.selected_index = None

        try:
            self._clear_rays()
            self._rebuild_system()
        except Exception as exc:
            self.model.components.insert(
                removed_index,
                removed,
            )
            self.selected_index = removed_index
            self._rebuild_system()
            messagebox.showerror(
                "Invalid component removal",
                f"Invalid component removal: {exc}",
            )
            return

        self._show_component_panel()
        self._render_scene()
        self._log(
            f"Removed component {removed.name}."
        )

    def _save_architecture(
        self,
    ):
        path = filedialog.asksaveasfilename(
            title="Save architecture",
            defaultextension=".json",
            filetypes=[
                ("JSON files", "*.json"),
            ],
        )

        if not path:
            return

        self._run_logged(
            lambda: self.model.save_json(path),
            success_message=(
                f"Architecture saved to {path}."
            ),
        )

    def _load_architecture(
        self,
    ):
        path = filedialog.askopenfilename(
            title="Load architecture",
            filetypes=[
                ("JSON files", "*.json"),
            ],
        )

        if not path:
            return

        def action():
            model = OpticalArchitectureModel.load_json(
                path
            )
            model.build_system()
            self.model = model
            self.selected_index = None
            self.move_component_var.set("")
            self.move_detector_1_var.set("")
            self.move_detector_2_var.set("")
            self._clear_rays()
            self._rebuild_system()
            self._show_component_panel()
            self._render_scene(
                preserve_limits=False
            )

        self._run_logged(
            action,
            success_message=(
                f"Architecture loaded from {path}."
            ),
        )

    def _find_component_by_name(
        self,
        name: str,
    ) -> ComponentSpec | None:
        for component in self.model.components:
            if component.name == name:
                return component
        return None

    def _render_scene(
        self,
        preserve_limits: bool = True,
    ):
        current_limits = None

        if preserve_limits and self.axes.has_data():
            current_limits = (
                self.axes.get_xlim(),
                self.axes.get_ylim(),
            )

        self.axes.clear()

        projected_limits = []
        for index, component in enumerate(
            self.model.components
        ):
            projected = _component_projection(
                component,
                self.current_view,
            )
            projected_limits.extend(
                projected["points"]
            )
            self._draw_component(
                index,
                component,
                projected,
            )

        source_xy = _project_point(
            self.model.source.position,
            self.current_view,
        )
        direction_xy = _project_vector(
            self.model.source.direction,
            self.current_view,
        )
        projected_limits.append(
            source_xy
        )

        self.axes.plot(
            [source_xy[0]],
            [source_xy[1]],
            marker="*",
            color="gold",
            markersize=12,
            zorder=40,
        )
        self.axes.annotate(
            "SRC",
            xy=source_xy,
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )
        self.axes.arrow(
            source_xy[0],
            source_xy[1],
            direction_xy[0] * 10.0,
            direction_xy[1] * 10.0,
            width=0.15,
            color="goldenrod",
            length_includes_head=True,
            zorder=30,
        )

        if self.current_rays is not None:
            for ray in self.current_rays:
                path = np.asarray(
                    ray.path
                )
                i, j = VIEW_MAP[
                    self.current_view
                ]
                self.axes.plot(
                    path[:, i],
                    path[:, j],
                    color="black",
                    alpha=0.4,
                    linewidth=1.0,
                    zorder=5,
                )
                projected_limits.extend(
                    path[:, [i, j]]
                )

        self.axes.grid(True)
        self.axes.set_aspect("equal")
        self.axes.set_title(
            self.current_view.upper()
        )
        self.axes.set_xlabel(
            self.current_view[0].upper() + " [mm]"
        )
        self.axes.set_ylabel(
            self.current_view[1].upper() + " [mm]"
        )

        if current_limits is not None:
            self.axes.set_xlim(
                current_limits[0]
            )
            self.axes.set_ylim(
                current_limits[1]
            )
        else:
            self._autoscale(
                projected_limits
            )

        self.canvas.draw_idle()

    def _draw_component(
        self,
        index: int,
        spec: ComponentSpec,
        projected: dict,
    ):
        color = COMPONENT_COLORS[
            spec.component_type
        ]
        selected = index == self.selected_index
        linewidth = 3.0 if selected else 2.0
        alpha = 0.25 if selected else 0.15

        if projected["kind"] == "volume":
            for face in projected["faces"]:
                self.axes.fill(
                    face[:, 0],
                    face[:, 1],
                    color=color,
                    alpha=alpha,
                    zorder=10,
                )
                self.axes.plot(
                    face[:, 0],
                    face[:, 1],
                    color=color,
                    linewidth=linewidth,
                    zorder=20,
                )

            for segment in projected["edges"]:
                self.axes.plot(
                    segment[:, 0],
                    segment[:, 1],
                    color=color,
                    linewidth=1.0,
                    alpha=0.8,
                    zorder=15,
                )
        else:
            outline = projected["outline"]
            self.axes.plot(
                outline[:, 0],
                outline[:, 1],
                color=color,
                linewidth=linewidth,
                zorder=20,
            )

        label_weight = (
            "bold"
            if selected
            else "normal"
        )
        self.axes.text(
            projected["center"][0],
            projected["center"][1],
            spec.name,
            ha="center",
            fontsize=8,
            fontweight=label_weight,
            zorder=30,
        )

    def _autoscale(
        self,
        points,
    ):
        if not points:
            self.axes.set_xlim(-60, 60)
            self.axes.set_ylim(-60, 60)
            return

        data = np.asarray(points)
        x_min = float(
            np.min(data[:, 0])
        )
        x_max = float(
            np.max(data[:, 0])
        )
        y_min = float(
            np.min(data[:, 1])
        )
        y_max = float(
            np.max(data[:, 1])
        )

        dx = max(
            x_max - x_min,
            20.0,
        )
        dy = max(
            y_max - y_min,
            20.0,
        )

        self.axes.set_xlim(
            x_min - 0.1 * dx,
            x_max + 0.1 * dx,
        )
        self.axes.set_ylim(
            y_min - 0.1 * dy,
            y_max + 0.1 * dy,
        )

    def _on_button_press(
        self,
        event,
    ):
        if event.inaxes != self.axes or event.button != 1:
            return

        if event.xdata is None or event.ydata is None:
            return

        picked = self._pick_component(
            event.xdata,
            event.ydata,
        )

        if picked is None:
            return

        self.selected_index = picked
        self._show_component_panel()
        self._render_scene()

        shift_pressed = (
            event.key is not None
            and "shift" in event.key.lower()
        )

        spec = self.model.components[
            self.selected_index
        ]
        self.drag_state = {
            "mode": (
                "rotate"
                if shift_pressed
                else "move"
            ),
            "index": self.selected_index,
            "start_xy": (
                event.xdata,
                event.ydata,
            ),
            "start_position": list(
                spec.position
            ),
            "start_rotation": list(
                spec.rotation_deg
            ),
        }

    def _on_mouse_move(
        self,
        event,
    ):
        if self.drag_state is None:
            return

        if event.inaxes != self.axes:
            return

        if event.xdata is None or event.ydata is None:
            return

        spec = self.model.components[
            self.drag_state["index"]
        ]
        i, j = VIEW_MAP[
            self.current_view
        ]

        if self.drag_state["mode"] == "move":
            previous_position = list(
                spec.position
            )
            start_position = list(
                self.drag_state[
                    "start_position"
                ]
            )
            dx = event.xdata - self.drag_state[
                "start_xy"
            ][0]
            dy = event.ydata - self.drag_state[
                "start_xy"
            ][1]
            start_position[i] = _snap_drag_coordinate(
                start_position[i] + dx
            )
            start_position[j] = _snap_drag_coordinate(
                start_position[j] + dy
            )
            spec.position = start_position
        else:
            previous_rotation = list(
                spec.rotation_deg
            )
            axis = _screen_rotation_axis(
                self.current_view
            )
            delta = (
                event.xdata
                - self.drag_state["start_xy"][0]
                + event.ydata
                - self.drag_state["start_xy"][1]
            )
            rotation = list(
                self.drag_state[
                    "start_rotation"
                ]
            )
            rotation[axis] = round(
                rotation[axis] + delta
            )
            spec.rotation_deg = rotation

        self._clear_rays()
        try:
            self._rebuild_system()
        except Exception as exc:
            if self.drag_state["mode"] == "move":
                spec.position = previous_position
            else:
                spec.rotation_deg = previous_rotation
            self._rebuild_system()
            self._log(
                f"Drag update rejected: {exc}"
            )
            return
        if self.panel_mode == "component":
            self._update_component_vars_from_spec(
                spec
            )
        self._render_scene()

    def _on_button_release(
        self,
        _event,
    ):
        self.drag_state = None

    def _on_scroll(
        self,
        event,
    ):
        if event.inaxes != self.axes:
            return

        if event.xdata is None or event.ydata is None:
            return

        scale = (
            0.9
            if event.button == "up"
            else 1.1
        )

        x_min, x_max = self.axes.get_xlim()
        y_min, y_max = self.axes.get_ylim()

        new_width = (x_max - x_min) * scale
        new_height = (y_max - y_min) * scale
        width = x_max - x_min
        height = y_max - y_min

        if width == 0 or height == 0:
            return

        x_ratio = (
            event.xdata - x_min
        ) / width
        y_ratio = (
            event.ydata - y_min
        ) / height

        self.axes.set_xlim(
            event.xdata - new_width * x_ratio,
            event.xdata + new_width * (1 - x_ratio),
        )
        self.axes.set_ylim(
            event.ydata - new_height * y_ratio,
            event.ydata + new_height * (1 - y_ratio),
        )
        self.canvas.draw_idle()

    def _pick_component(
        self,
        x: float,
        y: float,
    ) -> int | None:
        tolerance = self._pick_tolerance()
        best_index = None
        best_score = np.inf

        for index in range(
            len(self.model.components) - 1,
            -1,
            -1,
        ):
            spec = self.model.components[
                index
            ]
            projected = _component_projection(
                spec,
                self.current_view,
            )
            score = _projection_distance(
                projected,
                x,
                y,
            )

            if score < best_score:
                best_score = score
                best_index = index

        if best_score <= tolerance:
            return best_index

        return None

    def _pick_tolerance(
        self,
    ) -> float:
        x_min, x_max = self.axes.get_xlim()
        y_min, y_max = self.axes.get_ylim()
        return max(
            1.5,
            0.02 * max(
                x_max - x_min,
                y_max - y_min,
            ),
        )

    def _update_component_vars_from_spec(
        self,
        spec: ComponentSpec,
    ):
        if not self.component_vars:
            return

        self.component_vars["name"].set(
            spec.name
        )

        for axis, value in zip(
            AXIS_NAMES,
            spec.position,
        ):
            self.component_vars[
                f"position_{axis}"
            ].set(
                round(float(value), 6)
            )

        for axis, value in zip(
            AXIS_NAMES,
            spec.rotation_deg,
        ):
            self.component_vars[
                f"rotation_{axis}"
            ].set(
                round(float(value), 6)
            )

        for key, _, value in self._component_scalar_fields(
            spec
        ):
            if key in self.component_vars:
                self.component_vars[key].set(
                    round(float(value), 6)
                )

        for key, _, value in self._component_material_fields(
            spec
        ):
            if key in self.component_vars:
                self.component_vars[key].set(
                    value
                )
            numeric_key = (
                f"{key}_n"
            )
            if numeric_key in self.component_vars:
                self.component_vars[
                    numeric_key
                ].set(
                    round(
                        float(
                            getattr(
                                spec,
                                numeric_key,
                            )
                        ),
                        6,
                    )
                )


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


def _component_projection(
    spec: ComponentSpec,
    view: str,
) -> dict:
    rotation = _rotation_from_degrees(
        spec.rotation_deg
    )
    position = np.asarray(
        spec.position,
        dtype=float,
    )
    i, j = VIEW_MAP[view]

    if spec.component_type in THICK_COMPONENTS:
        front_world, back_world = _thick_component_faces(
            position,
            rotation,
            spec.width,
            spec.height,
            spec.thickness,
        )
        center = (
            position
            + rotation.apply(
                np.array(
                    [0.0, 0.0, spec.thickness / 2]
                )
            )
        )

        front_xy = front_world[:, [i, j]]
        back_xy = back_world[:, [i, j]]
        edges = []

        for front_point, back_point in zip(
            front_world[:-1],
            back_world[:-1],
        ):
            edges.append(
                np.array([
                    front_point[[i, j]],
                    back_point[[i, j]],
                ])
            )

        points = list(front_xy) + list(back_xy)

        return {
            "kind": "volume",
            "faces": [front_xy, back_xy],
            "edges": edges,
            "center": center[[i, j]],
            "points": points,
        }

    outline = _local_rectangle(
        spec.width,
        spec.height,
        z=0.0,
    )
    world_outline = np.array([
        rotation.apply(point) + position
        for point in outline
    ])
    outline_xy = world_outline[:, [i, j]]

    return {
        "kind": "outline",
        "outline": outline_xy,
        "center": position[[i, j]],
        "points": list(outline_xy),
    }


def _local_rectangle(
    width: float,
    height: float,
    z: float,
) -> np.ndarray:
    half_width = width / 2
    half_height = height / 2

    return np.array([
        [-half_width, -half_height, z],
        [half_width, -half_height, z],
        [half_width, half_height, z],
        [-half_width, half_height, z],
        [-half_width, -half_height, z],
    ])


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


def _thick_component_faces(
    front_position: np.ndarray,
    rotation: Rotation,
    width: float,
    height: float,
    thickness: float,
) -> tuple[np.ndarray, np.ndarray]:
    front = _local_rectangle(
        width,
        height,
        z=0.0,
    )
    back = _local_rectangle(
        width,
        height,
        z=thickness,
    )

    front_world = np.array([
        rotation.apply(point) + front_position
        for point in front
    ])
    back_world = np.array([
        rotation.apply(point) + front_position
        for point in back
    ])

    return front_world, back_world


def launch_optical_editor(
    model: OpticalArchitectureModel | None = None,
):
    if TK_IMPORT_ERROR is not None:
        raise RuntimeError(
            "tkinter is required to launch the optical editor."
        ) from TK_IMPORT_ERROR

    root = tk.Tk()
    OpticalEditorApp(
        root=root,
        model=model,
    )
    root.mainloop()


# ==========================================================
# Lightweight Matplotlib editor API
# ==========================================================

SOURCE_COLOR = "gold"
SELECTED_COLOR = "black"


@dataclass
class OperationProgress:
    operation: str | None = None
    state: str = "idle"
    value: float | None = None
    message: str = "Idle"
    history: list[tuple[str, float | None, str]] = field(
        default_factory=list
    )

    def _remember(
        self,
    ):
        self.history.append(
            (
                self.state,
                self.value,
                self.message,
            )
        )

    def start(
        self,
        operation,
        determinate,
        message,
    ):
        self.operation = operation
        self.state = "running"
        self.value = (
            0.0
            if determinate
            else None
        )
        self.message = message
        self._remember()

    def update(
        self,
        value=None,
        message=None,
    ):
        if self.state == "idle":
            self.state = "running"

        self.value = (
            None
            if value is None
            else min(
                1.0,
                max(
                    0.0,
                    float(value),
                ),
            )
        )

        if message is not None:
            self.message = message

        self._remember()

    def finish(
        self,
        success=True,
        message=None,
    ):
        self.state = (
            "success"
            if success
            else "error"
        )

        if success:
            self.value = 1.0

        if message is not None:
            self.message = message

        self._remember()

    def reset(
        self,
    ):
        self.operation = None
        self.state = "idle"
        self.value = None
        self.message = "Idle"
        self._remember()


@dataclass
class SelectionState:
    selected_object: object | None = None
    interaction_mode: str = "select"


class OpticalEditor:

    def __init__(
        self,
        target,
        view="xz",
        fig=None,
    ):
        self.target = target
        self.system = getattr(
            target,
            "system",
            target,
        )
        self.view = view
        self.progress = OperationProgress()
        self.selection = SelectionState()
        self.current_rays = None
        self.material_specs = (
            discover_material_specs()
        )
        self.optic_specs = (
            discover_optic_specs()
        )
        self.inspector_sections = []
        self._widget_axes = []
        self._widgets = {}
        self._source_artist = None
        self._source_handle_artist = None
        self._selected_outline = None
        self._drag_anchor = None
        self._drag_source_position = None

        self.figure = (
            fig
            if fig is not None
            else plt.figure(
                figsize=(12, 7)
            )
        )

        grid = self.figure.add_gridspec(
            2,
            2,
            width_ratios=[3.2, 1.4],
            height_ratios=[20, 2],
            wspace=0.12,
            hspace=0.12,
        )

        self.plot_ax = self.figure.add_subplot(
            grid[:, 0]
        )
        self.panel_ax = self.figure.add_subplot(
            grid[0, 1]
        )
        self.progress_ax = (
            self.figure.add_subplot(
                grid[1, 1]
            )
        )

        self._configure_axes()
        self._build_buttons()
        self.select_object(
            self.system.source
        )
        self.refresh()

        self._cids = [
            self.figure.canvas.mpl_connect(
                "button_press_event",
                self.on_mouse_press,
            ),
            self.figure.canvas.mpl_connect(
                "motion_notify_event",
                self.on_mouse_move,
            ),
            self.figure.canvas.mpl_connect(
                "button_release_event",
                self.on_mouse_release,
            ),
            self.figure.canvas.mpl_connect(
                "key_press_event",
                self.on_key_press,
            ),
        ]

    # ==================================================
    # Layout
    # ==================================================

    def _configure_axes(
        self,
    ):
        self.panel_ax.set_xticks([])
        self.panel_ax.set_yticks([])
        self.panel_ax.set_facecolor(
            "#f8f8f8"
        )
        self.panel_ax.set_title(
            "Inspector",
            loc="left",
        )

        self.progress_ax.set_xticks([])
        self.progress_ax.set_yticks([])
        self.progress_ax.set_xlim(
            0,
            1,
        )
        self.progress_ax.set_ylim(
            0,
            1,
        )
        self.progress_ax.set_title(
            "Progress",
            loc="left",
            fontsize=10,
        )

    def _build_buttons(
        self,
    ):
        trace_ax = self.figure.add_axes(
            [0.78, 0.12, 0.08, 0.045]
        )
        move_ax = self.figure.add_axes(
            [0.88, 0.12, 0.08, 0.045]
        )

        self.trace_button = Button(
            trace_ax,
            "Trace",
        )
        self.move_button = Button(
            move_ax,
            "Move",
        )

        self.trace_button.on_clicked(
            lambda _event: self.run_trace()
        )
        self.move_button.on_clicked(
            lambda _event: self.run_move()
        )

    # ==================================================
    # Inspector
    # ==================================================

    def beam_source(
        self,
    ):
        return self.system.source

    def get_beam_parameters(
        self,
    ):
        source = self.beam_source()

        if source is None:
            return {}

        return source.get_beam_parameters()

    def get_catalog(
        self,
    ):
        return {
            "materials": [
                spec.label
                for spec in self.material_specs
            ],
            "optics": [
                spec.label
                for spec in self.optic_specs
            ],
        }

    def select_object(
        self,
        obj,
    ):
        self.selection.selected_object = (
            obj
        )
        self.selection.interaction_mode = (
            "select"
        )
        self._build_inspector()

    def _build_inspector(
        self,
    ):
        for axis in self._widget_axes:
            axis.remove()

        self._widget_axes = []
        self._widgets = {}
        self.inspector_sections = []

        self.panel_ax.clear()
        self.panel_ax.set_xticks([])
        self.panel_ax.set_yticks([])
        self.panel_ax.set_facecolor(
            "#f8f8f8"
        )
        self.panel_ax.set_title(
            "Inspector",
            loc="left",
        )

        selected = (
            self.selection.selected_object
        )

        if selected is None:
            title = "No selection"
        else:
            title = getattr(
                selected,
                "name",
                selected.__class__.__name__,
            )

        self.panel_ax.text(
            0.03,
            0.96,
            title,
            transform=
                self.panel_ax.transAxes,
            va="top",
            fontsize=11,
            fontweight="bold",
        )

        sections = []

        if selected is not None:
            sections.append(
                (
                    "Object Parameters",
                    self._object_parameters(
                        selected
                    ),
                )
            )

        if (
            selected is self.system.source
            or selected in self.system.detectors
        ):
            sections.append(
                (
                    "Beam / Source Parameters",
                    self._beam_parameters(),
                )
            )

        sections.append(
            (
                "Available Materials",
                self.get_catalog()[
                    "materials"
                ],
            )
        )
        sections.append(
            (
                "Available Optics",
                self.get_catalog()[
                    "optics"
                ],
            )
        )

        self.inspector_sections = [
            {
                "title": title,
                "items": items,
            }
            for title, items in sections
        ]

        self._render_sections(
            sections
        )

    def _object_parameters(
        self,
        obj,
    ):
        items = []

        transform = getattr(
            obj,
            "transform",
            None,
        )

        if transform is not None:
            for axis, value in zip(
                "xyz",
                transform.position,
            ):
                items.append(
                    (
                        f"position_{axis}",
                        float(value),
                        partial(
                            self._set_object_position,
                            obj,
                            axis,
                        ),
                    )
                )

        if hasattr(
            obj,
            "width",
        ):
            items.append(
                (
                    "width",
                    float(obj.width),
                    None,
                )
            )

        if hasattr(
            obj,
            "height",
        ):
            items.append(
                (
                    "height",
                    float(obj.height),
                    None,
                )
            )

        return items

    def _beam_parameters(
        self,
    ):
        source = self.beam_source()

        if source is None:
            return []

        items = []

        for axis, value in zip(
            "xyz",
            source.position,
        ):
            items.append(
                (
                    f"position_{axis}",
                    float(value),
                    partial(
                        self.set_beam_parameter,
                        f"position_{axis}",
                    ),
                )
            )

        for axis, value in zip(
            "xyz",
            source.direction,
        ):
            items.append(
                (
                    f"direction_{axis}",
                    float(value),
                    partial(
                        self.set_beam_parameter,
                        f"direction_{axis}",
                    ),
                )
            )

        items.append(
            (
                "wavelength",
                float(
                    source.wavelength
                ),
                partial(
                    self.set_beam_parameter,
                    "wavelength",
                ),
            )
        )

        if hasattr(
            source,
            "radius",
        ):
            items.append(
                (
                    "radius",
                    float(
                        source.radius
                    ),
                    partial(
                        self.set_beam_parameter,
                        "radius",
                    ),
                )
            )

        if hasattr(
            source,
            "n_rays",
        ):
            items.append(
                (
                    "n_rays",
                    int(
                        source.n_rays
                    ),
                    partial(
                        self.set_beam_parameter,
                        "n_rays",
                    ),
                )
            )

        return items

    def _render_sections(
        self,
        sections,
    ):
        figure = self.figure
        panel_bounds = (
            self.panel_ax
            .get_position()
            .bounds
        )

        left, bottom, width, height = (
            panel_bounds
        )
        y_cursor = bottom + height * 0.90

        for title, items in sections:
            self.panel_ax.text(
                0.03,
                (
                    (y_cursor - bottom)
                    / height
                ),
                title,
                transform=
                    self.panel_ax.transAxes,
                va="top",
                fontsize=9,
                fontweight="bold",
            )
            y_cursor -= 0.03

            if items and isinstance(
                items[0],
                tuple,
            ):
                for name, value, callback in items:
                    y_cursor -= 0.048
                    if y_cursor <= bottom:
                        break

                    box_ax = figure.add_axes(
                        [
                            left + width * 0.03,
                            y_cursor,
                            width * 0.94,
                            0.035,
                        ]
                    )
                    self._widget_axes.append(
                        box_ax
                    )

                    label = name.replace(
                        "_",
                        " ",
                    ).title()

                    widget = TextBox(
                        box_ax,
                        label,
                        initial=str(value),
                    )

                    if callback is not None:
                        widget.on_submit(
                            callback
                        )

                    self._widgets[name] = (
                        widget
                    )
            else:
                preview = ", ".join(
                    map(
                        str,
                        items[:8],
                    )
                )
                self.panel_ax.text(
                    0.03,
                    (
                        (y_cursor - bottom)
                        / height
                    ),
                    preview,
                    transform=
                        self.panel_ax.transAxes,
                    va="top",
                    fontsize=8,
                    wrap=True,
                )
                y_cursor -= 0.08

            y_cursor -= 0.02

        self.figure.canvas.draw_idle()

    def _set_object_position(
        self,
        obj,
        axis,
        raw_value,
    ):
        if not hasattr(
            obj,
            "transform",
        ):
            return

        axis_index = "xyz".index(
            axis
        )
        position = (
            obj.transform.position.copy()
        )
        position[axis_index] = float(
            raw_value
        )
        obj.transform.position = (
            position
        )
        self.refresh()

    def set_beam_parameter(
        self,
        name,
        raw_value,
    ):
        source = self.beam_source()

        if source is None:
            raise RuntimeError(
                "No source defined."
            )

        if name.startswith(
            "position_"
        ):
            axis = "xyz".index(
                name[-1]
            )
            position = (
                source.position.copy()
            )
            position[axis] = float(
                raw_value
            )
            source.set_position(
                position
            )
        elif name.startswith(
            "direction_"
        ):
            axis = "xyz".index(
                name[-1]
            )
            direction = (
                source.direction.copy()
            )
            direction[axis] = float(
                raw_value
            )
            source.set_direction(
                direction
            )
        elif name == "wavelength":
            source.set_wavelength(
                raw_value
            )
        elif name == "radius":
            source.set_radius(
                raw_value
            )
        elif name == "n_rays":
            source.set_n_rays(
                raw_value
            )
        else:
            raise KeyError(
                f"Unknown beam parameter "
                f"'{name}'"
            )

        self.refresh()

    # ==================================================
    # Rendering
    # ==================================================

    def refresh(
        self,
    ):
        self._draw_plot()
        self._draw_progress()
        self.figure.canvas.draw_idle()

    def _draw_plot(
        self,
    ):
        self.plot_ax.clear()

        for element in self.system.tracer.elements:
            plot_element(
                element,
                view=self.view,
                ax=self.plot_ax,
            )

        if self.current_rays is not None:
            for ray in self.current_rays:
                plot_ray(
                    ray,
                    view=self.view,
                    ax=self.plot_ax,
                )

        self._draw_source()
        self._highlight_selection()

        self.plot_ax.grid(True)
        self.plot_ax.set_aspect(
            "equal"
        )
        self.plot_ax.set_title(
            self.view.upper()
        )

    def _draw_source(
        self,
    ):
        source = self.beam_source()

        if source is None:
            return

        i, j = VIEW_MAP[
            self.view
        ]

        origin = np.asarray(
            [
                source.position[i],
                source.position[j],
            ]
        )

        direction = np.asarray(
            [
                source.direction[i],
                source.direction[j],
            ]
        )

        norm = np.linalg.norm(
            direction
        )

        if norm == 0:
            direction = np.array(
                [1.0, 0.0]
            )
        else:
            direction = (
                direction / norm
            )

        scale = self._source_handle_length()
        handle = origin + scale * direction

        self._source_artist, = (
            self.plot_ax.plot(
                [origin[0]],
                [origin[1]],
                marker="o",
                markersize=9,
                color=SOURCE_COLOR,
                mec=SELECTED_COLOR,
                zorder=40,
            )
        )

        self.plot_ax.plot(
            [origin[0], handle[0]],
            [origin[1], handle[1]],
            color=SOURCE_COLOR,
            linewidth=2,
            zorder=39,
        )

        self._source_handle_artist = (
            Circle(
                handle,
                radius=scale * 0.12,
                facecolor="white",
                edgecolor=SOURCE_COLOR,
                linewidth=2,
                zorder=41,
            )
        )
        self.plot_ax.add_patch(
            self._source_handle_artist
        )
        self.plot_ax.text(
            origin[0],
            origin[1],
            "Source",
            fontsize=8,
            ha="left",
            va="bottom",
            zorder=42,
        )

    def _highlight_selection(
        self,
    ):
        selected = (
            self.selection.selected_object
        )

        if selected is None:
            return

        if selected is self.system.source:
            return

        if hasattr(
            selected,
            "get_outline",
        ):
            outline = (
                selected.get_outline()
            )
            i, j = VIEW_MAP[
                self.view
            ]
            self.plot_ax.plot(
                outline[:, i],
                outline[:, j],
                color=SELECTED_COLOR,
                linewidth=4,
                alpha=0.35,
                zorder=35,
            )

    def _draw_progress(
        self,
    ):
        self.progress_ax.clear()
        self.progress_ax.set_xticks([])
        self.progress_ax.set_yticks([])
        self.progress_ax.set_xlim(
            0,
            1,
        )
        self.progress_ax.set_ylim(
            0,
            1,
        )
        self.progress_ax.set_title(
            "Progress",
            loc="left",
            fontsize=10,
        )

        colors = {
            "idle": "#d9d9d9",
            "running": "#3776ab",
            "success": "#2a9d4b",
            "error": "#c0392b",
        }

        background = Rectangle(
            (0.02, 0.25),
            0.96,
            0.35,
            facecolor="#efefef",
            edgecolor="#cccccc",
        )
        self.progress_ax.add_patch(
            background
        )

        value = self.progress.value

        if self.progress.state == "running" and value is None:
            fill = Rectangle(
                (0.02, 0.25),
                0.96,
                0.35,
                facecolor="none",
                edgecolor=colors[
                    self.progress.state
                ],
                hatch="////",
                linewidth=1.5,
            )
        else:
            fill = Rectangle(
                (0.02, 0.25),
                0.96 * (
                    0.0
                    if value is None
                    else value
                ),
                0.35,
                facecolor=colors[
                    self.progress.state
                ],
                edgecolor=colors[
                    self.progress.state
                ],
            )

        self.progress_ax.add_patch(
            fill
        )
        self.progress_ax.text(
            0.02,
            0.82,
            self.progress.message,
            fontsize=9,
            va="top",
        )

    # ==================================================
    # Actions
    # ==================================================

    def reset_progress(
        self,
    ):
        self.progress.reset()
        self.refresh()

    def run_trace(
        self,
    ):
        determinate = hasattr(
            self.beam_source(),
            "n_rays",
        )
        self.progress.start(
            "trace",
            determinate=determinate,
            message="Tracing…",
        )
        self.refresh()

        try:
            runner = getattr(
                self.target,
                "trace",
                self.system.trace,
            )
            self.current_rays = runner(
                progress_callback=
                    self._progress_callback
            )
        except Exception as exc:
            self.progress.finish(
                success=False,
                message=str(exc),
            )
            self.refresh()
            raise

        self.progress.finish(
            success=True,
            message="Trace complete",
        )
        self.refresh()
        return self.current_rays

    def run_move(
        self,
        mirror_positions=None,
    ):
        if not hasattr(
            self.target,
            "scan_mirror",
        ):
            raise RuntimeError(
                "Target does not support move."
            )

        if mirror_positions is None:
            wavelength = (
                self.beam_source().wavelength
                if self.beam_source()
                is not None
                else 632.8e-9
            )
            mirror_positions = np.linspace(
                -wavelength,
                wavelength,
                9,
            )

        self.progress.start(
            "move",
            determinate=True,
            message="Moving…",
        )
        self.refresh()

        try:
            result = self.target.scan_mirror(
                mirror_positions,
                progress_callback=
                    self._progress_callback
            )
        except Exception as exc:
            self.progress.finish(
                success=False,
                message=str(exc),
            )
            self.refresh()
            raise

        self.progress.finish(
            success=True,
            message="Move complete",
        )
        self.refresh()
        return result

    def _progress_callback(
        self,
        value=None,
        message=None,
        current=None,
        total=None,
    ):
        self.progress.update(
            value=value,
            message=message,
        )
        self._draw_progress()
        self.figure.canvas.draw_idle()

    # ==================================================
    # Mouse interaction
    # ==================================================

    def on_mouse_press(
        self,
        event,
    ):
        if event.button != 1:
            return

        if event.inaxes != self.plot_ax:
            return

        point = self._event_point(
            event
        )

        if point is None:
            return

        source = self.beam_source()

        if source is not None:
            if self._is_near_source_handle(
                point
            ):
                self.select_object(
                    source
                )
                self.selection.interaction_mode = (
                    "rotate_source"
                )
                return

            if self._is_near_source_origin(
                point
            ):
                self.select_object(
                    source
                )
                self.selection.interaction_mode = (
                    "move_source"
                )
                self._drag_anchor = point
                self._drag_source_position = (
                    source.position.copy()
                )
                return

        selected = self._find_selected_object(
            point
        )
        if selected is not None:
            self.select_object(
                selected
            )

    def on_mouse_move(
        self,
        event,
    ):
        if event.inaxes != self.plot_ax:
            return

        point = self._event_point(
            event
        )

        if point is None:
            return

        source = self.beam_source()

        if (
            source is None
            or
            self.selection.selected_object
            is not source
        ):
            return

        if (
            self.selection.interaction_mode
            == "move_source"
        ):
            delta_2d = (
                point
                - self._drag_anchor
            )
            delta_3d = np.zeros(3)
            i, j = VIEW_MAP[
                self.view
            ]
            delta_3d[i] = delta_2d[0]
            delta_3d[j] = delta_2d[1]
            source.set_position(
                self._drag_source_position
                + delta_3d
            )
            self.refresh()
        elif (
            self.selection.interaction_mode
            == "rotate_source"
        ):
            i, j = VIEW_MAP[
                self.view
            ]
            origin = np.array(
                [
                    source.position[i],
                    source.position[j],
                ]
            )
            direction_2d = point - origin
            norm = np.linalg.norm(
                direction_2d
            )

            if norm == 0:
                return

            direction = np.zeros(3)
            direction[i] = (
                direction_2d[0]
                / norm
            )
            direction[j] = (
                direction_2d[1]
                / norm
            )
            source.set_direction(
                direction
            )
            self.refresh()

    def on_mouse_release(
        self,
        event,
    ):
        self.selection.interaction_mode = (
            "select"
        )
        self._drag_anchor = None
        self._drag_source_position = None

    def on_key_press(
        self,
        event,
    ):
        key = getattr(
            event,
            "key",
            None,
        )

        if key is None:
            return

        if key == "tab":
            self._mark_event_handled(
                event
            )
            self._cycle_selection(
                1
            )
            return

        if key == "shift+tab":
            self._mark_event_handled(
                event
            )
            self._cycle_selection(
                -1
            )
            return

        if key == "t":
            self.run_trace()
            return

        if key == "m":
            self.run_move()
            return

        source = self.beam_source()

        if (
            source is None
            or
            self.selection.selected_object
            is not source
        ):
            return

        if key in {
            "left",
            "right",
            "up",
            "down",
        }:
            self._nudge_source(
                key
            )
            return

        if key in {
            "shift+left",
            "shift+right",
            "shift+up",
            "shift+down",
        }:
            self._rotate_source_with_key(
                key.split(
                    "+",
                    1,
                )[1]
            )

    def _event_point(
        self,
        event,
    ):
        if event.xdata is None or event.ydata is None:
            return None

        return np.array(
            [
                event.xdata,
                event.ydata,
            ],
            dtype=float,
        )

    def _mark_event_handled(
        self,
        event,
    ):
        setattr(
            event,
            "handled",
            True,
        )

    def _source_handle_length(
        self,
    ):
        source = self.beam_source()

        if source is None:
            return 1.0

        scale = 0.25

        if hasattr(
            source,
            "radius",
        ):
            scale = max(
                scale,
                3.0 * source.radius,
            )

        return scale

    def _selectable_objects(
        self,
    ):
        objects = list(
            self.system.tracer.elements
        )

        if self.beam_source() is not None:
            objects.insert(
                0,
                self.beam_source()
            )

        return objects

    def _cycle_selection(
        self,
        step,
    ):
        selectable = (
            self._selectable_objects()
        )

        if not selectable:
            return

        selected = (
            self.selection.selected_object
        )

        try:
            index = selectable.index(
                selected
            )
        except ValueError:
            index = -1

        self.select_object(
            selectable[
                (
                    index + step
                )
                % len(selectable)
            ]
        )
        self.refresh()

    def _nudge_source(
        self,
        key,
    ):
        step = max(
            0.1,
            self._source_handle_length()
            * 0.25,
        )
        delta_2d = {
            "left": np.array(
                [-step, 0.0]
            ),
            "right": np.array(
                [step, 0.0]
            ),
            "up": np.array(
                [0.0, step]
            ),
            "down": np.array(
                [0.0, -step]
            ),
        }[key]

        delta_3d = np.zeros(3)
        i, j = VIEW_MAP[
            self.view
        ]
        delta_3d[i] = delta_2d[0]
        delta_3d[j] = delta_2d[1]

        self.beam_source().set_position(
            self.beam_source().position
            + delta_3d
        )
        self.refresh()

    def _rotate_source_with_key(
        self,
        key,
    ):
        direction_2d = {
            "left": np.array(
                [-1.0, 0.0]
            ),
            "right": np.array(
                [1.0, 0.0]
            ),
            "up": np.array(
                [0.0, 1.0]
            ),
            "down": np.array(
                [0.0, -1.0]
            ),
        }[key]

        direction = np.zeros(3)
        i, j = VIEW_MAP[
            self.view
        ]
        direction[i] = direction_2d[0]
        direction[j] = direction_2d[1]
        self.beam_source().set_direction(
            direction
        )
        self.refresh()

    def _source_origin_2d(
        self,
    ):
        source = self.beam_source()
        i, j = VIEW_MAP[
            self.view
        ]
        return np.array(
            [
                source.position[i],
                source.position[j],
            ]
        )

    def _source_handle_2d(
        self,
    ):
        source = self.beam_source()
        i, j = VIEW_MAP[
            self.view
        ]
        direction = np.array(
            [
                source.direction[i],
                source.direction[j],
            ]
        )
        norm = np.linalg.norm(
            direction
        )
        if norm == 0:
            direction = np.array(
                [1.0, 0.0]
            )
        else:
            direction = (
                direction / norm
            )
        return (
            self._source_origin_2d()
            +
            self._source_handle_length()
            * direction
        )

    def _hit_tolerance(
        self,
    ):
        return max(
            0.2,
            self._source_handle_length()
            * 0.35,
        )

    def _is_near_source_origin(
        self,
        point,
    ):
        return (
            np.linalg.norm(
                point
                - self._source_origin_2d()
            )
            <= self._hit_tolerance()
        )

    def _is_near_source_handle(
        self,
        point,
    ):
        return (
            np.linalg.norm(
                point
                - self._source_handle_2d()
            )
            <= self._hit_tolerance()
        )

    def _find_selected_object(
        self,
        point,
    ):
        best_distance = np.inf
        best_object = None

        for element in (
            self._selectable_objects()
        ):
            distance = self._distance_to_element(
                element,
                point,
            )

            if distance < best_distance:
                best_distance = distance
                best_object = element

        if best_distance <= (
            self._hit_tolerance() * 2.0
        ):
            return best_object

        return None

    def _distance_to_element(
        self,
        element,
        point,
    ):
        if element is self.beam_source():
            return np.linalg.norm(
                point
                - self._source_origin_2d()
            )

        i, j = VIEW_MAP[
            self.view
        ]

        if hasattr(
            element,
            "get_outline",
        ):
            outline = (
                element.get_outline()
            )[:, [i, j]]
            if len(outline) < 2:
                if len(outline) == 1:
                    return np.linalg.norm(
                        point
                        - outline[0]
                    )
                center = np.array(
                    [
                        element.transform.position[i],
                        element.transform.position[j],
                    ]
                )
                return np.linalg.norm(
                    point - center
                )
            segments = list(
                zip(
                    outline[:-1],
                    outline[1:],
                )
            )
            if not np.allclose(
                outline[0],
                outline[-1],
            ):
                segments.append(
                    (
                        outline[-1],
                        outline[0],
                    )
                )
            return min(
                self._distance_to_segment(
                    point,
                    start,
                    end,
                )
                for start, end in segments
            )

        center = np.array(
            [
                element.transform.position[i],
                element.transform.position[j],
            ]
        )
        return np.linalg.norm(
            point - center
        )

    def _distance_to_segment(
        self,
        point,
        start,
        end,
    ):
        segment = end - start
        length_sq = np.dot(
            segment,
            segment,
        )

        if length_sq == 0:
            return np.linalg.norm(
                point - start
            )

        t = np.dot(
            point - start,
            segment,
        ) / length_sq
        t = min(
            1.0,
            max(
                0.0,
                t,
            ),
        )
        projection = (
            start + t * segment
        )
        return np.linalg.norm(
            point - projection
        )
