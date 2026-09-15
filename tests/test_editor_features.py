import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from architectures.classic_michelson import ClassicMichelson
from materials.constant_index import ConstantIndex
from optics.bundle_source import CircularBundleSource
from optics.source import Source
from plotting.editor import OpticalEditor


def build_michelson(
    source=None,
):
    system = ClassicMichelson(
        arm_length=10,
        compensator_distance=5,
        mirror_size=3,
        plate_size=4,
        plate_thickness=1,
        material=ConstantIndex(
            1.5,
            "BK7",
        ),
        detector_distance=6,
    )

    if source is None:
        source = CircularBundleSource(
            position=[-4, 0, 0],
            direction=[1, 0, 0],
            wavelength=632.8e-9,
            radius=0.2,
            n_rays=3,
        )

    system.set_source(
        source
    )
    return system


def make_event(
    editor,
    x,
    y,
):
    return SimpleNamespace(
        inaxes=editor.plot_ax,
        xdata=x,
        ydata=y,
        button=1,
    )


def test_detector_selection_exposes_shared_beam_parameters():
    michelson = build_michelson()
    editor = OpticalEditor(
        michelson
    )

    detector = michelson.detector1
    point = detector.get_outline().mean(
        axis=0
    )
    editor.on_mouse_press(
        make_event(
            editor,
            point[0],
            point[2],
        )
    )

    titles = [
        section["title"]
        for section in editor.inspector_sections
    ]

    assert (
        "Beam / Source Parameters"
        in titles
    )

    editor.set_beam_parameter(
        "radius",
        "0.55",
    )
    assert np.isclose(
        michelson.system.source.radius,
        0.55,
    )


def test_source_mouse_move_and_rotate():
    michelson = build_michelson(
        source=Source(
            position=[-4, 0, 0],
            direction=[1, 0, 0],
        )
    )
    editor = OpticalEditor(
        michelson
    )

    origin = editor._source_origin_2d()
    editor.on_mouse_press(
        make_event(
            editor,
            origin[0],
            origin[1],
        )
    )
    editor.on_mouse_move(
        make_event(
            editor,
            origin[0] + 1.5,
            origin[1] + 0.75,
        )
    )
    editor.on_mouse_release(
        make_event(
            editor,
            origin[0] + 1.5,
            origin[1] + 0.75,
        )
    )

    assert np.allclose(
        michelson.system.source.position,
        [-2.5, 0.0, 0.75],
    )

    handle = editor._source_handle_2d()

    editor.on_mouse_press(
        make_event(
            editor,
            handle[0],
            handle[1],
        )
    )
    editor.on_mouse_move(
        make_event(
            editor,
            michelson.system.source.position[0],
            michelson.system.source.position[2] + 2.0,
        )
    )
    editor.on_mouse_release(
        make_event(
            editor,
            michelson.system.source.position[0],
            michelson.system.source.position[2] + 2.0,
        )
    )

    assert np.allclose(
        michelson.system.source.direction,
        [0.0, 0.0, 1.0],
    )


def test_progress_state_transitions_for_trace_and_move():
    michelson = build_michelson()
    editor = OpticalEditor(
        michelson
    )

    assert editor.progress.state == "idle"

    rays = editor.run_trace()
    assert rays
    assert editor.progress.state == "success"
    assert any(
        state == "running"
        for state, _value, _message
        in editor.progress.history
    )

    editor.reset_progress()
    assert editor.progress.state == "idle"

    result = editor.run_move(
        np.linspace(
            -1e-7,
            1e-7,
            5,
        )
    )
    assert result["det1"].shape == (5,)
    assert editor.progress.state == "success"
    assert any(
        value is not None
        for state, value, _message
        in editor.progress.history
        if state == "running"
    )


def test_scan_progress_payloads_are_consistent():
    michelson = build_michelson()

    events = []

    def callback(
        value=None,
        message=None,
        current=None,
        total=None,
    ):
        events.append(
            (
                value,
                message,
                current,
                total,
            )
        )

    michelson.scan_mirror(
        np.linspace(
            -1e-7,
            1e-7,
            3,
        ),
        progress_callback=callback,
    )

    completed_events = [
        event
        for event in events
        if event[1]
        and event[1].startswith(
            "Completed move step"
        )
    ]

    assert completed_events[0] == (
        1 / 3,
        "Completed move step 1/3",
        1,
        3,
    )
    assert completed_events[-1] == (
        1.0,
        "Completed move step 3/3",
        3,
        3,
    )


def test_keyboard_controls_provide_source_accessibility():
    michelson = build_michelson(
        source=Source(
            position=[-4, 0, 0],
            direction=[1, 0, 0],
        )
    )
    editor = OpticalEditor(
        michelson
    )

    editor.on_key_press(
        SimpleNamespace(
            key="right"
        )
    )
    assert (
        michelson.system.source.position[0]
        > -4.0
    )

    editor.on_key_press(
        SimpleNamespace(
            key="shift+up"
        )
    )
    assert np.allclose(
        michelson.system.source.direction,
        [0.0, 0.0, 1.0],
    )

    editor.on_key_press(
        SimpleNamespace(
            key="tab"
        )
    )
    assert (
        editor.selection.selected_object
        is not michelson.system.source
    )


def test_run_trace_progress_events_are_monotonic_for_bundle_sources():
    michelson = build_michelson()
    editor = OpticalEditor(
        michelson
    )

    events = []
    original_callback = (
        editor._progress_callback
    )

    def capture(
        value=None,
        message=None,
        current=None,
        total=None,
    ):
        events.append(
            (
                value,
                message,
                current,
                total,
            )
        )
        original_callback(
            value=value,
            message=message,
            current=current,
            total=total,
        )

    editor._progress_callback = capture
    editor.run_trace()

    determinate_values = [
        value
        for value, _message, _current, _total
        in events
        if value is not None
    ]

    assert determinate_values
    assert determinate_values == sorted(
        determinate_values
    )
    assert determinate_values[-1] == 1.0


def test_progress_error_state_when_trace_fails():
    michelson = build_michelson(
        source=Source(
            position=[-4, 0, 0],
            direction=[1, 0, 0],
        )
    )
    michelson.system.source = None

    editor = OpticalEditor(
        michelson
    )

    with pytest.raises(
        RuntimeError,
        match="No source defined",
    ):
        editor.run_trace()

    assert editor.progress.state == "error"


def test_single_ray_trace_uses_indeterminate_progress():
    michelson = build_michelson(
        source=Source(
            position=[-4, 0, 0],
            direction=[1, 0, 0],
        )
    )
    editor = OpticalEditor(
        michelson
    )

    editor.run_trace()

    assert any(
        value is None
        for state, value, _message
        in editor.progress.history
        if state == "running"
    )
