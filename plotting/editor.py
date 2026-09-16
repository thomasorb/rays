from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
from matplotlib.widgets import TextBox
import numpy as np

from plotting.views import VIEW_MAP
from plotting.views import plot_element
from plotting.views import plot_ray
from registry import discover_material_specs
from registry import discover_optic_specs


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

        for detector in self.system.detectors:
            plot_element(
                detector,
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
            span = 1e-3
            if hasattr(
                self.target,
                "arm_length",
            ):
                span = max(
                    span,
                    abs(
                        float(
                            self.target.arm_length
                        )
                    )
                    * 1e-3,
                )
            mirror_positions = np.linspace(
                -span,
                span,
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
        objects.extend(
            self.system.detectors
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

            if (
                len(outline) >= 3
                and self._point_in_polygon(
                    point,
                    outline,
                )
            ):
                return 0.0

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

    def _point_in_polygon(
        self,
        point,
        polygon,
    ):
        x, y = point
        inside = False

        for start, end in zip(
            polygon,
            np.roll(
                polygon,
                -1,
                axis=0,
            ),
        ):
            x1, y1 = start
            x2, y2 = end

            intersects = (
                (y1 > y)
                != (y2 > y)
            )
            if not intersects:
                continue

            x_cross = x1 + (
                (y - y1)
                * (x2 - x1)
                / (y2 - y1)
            )
            if x <= x_cross:
                inside = not inside

        return inside

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
