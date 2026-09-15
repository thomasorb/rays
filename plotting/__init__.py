__all__ = [
    "OpticalArchitectureModel",
    "OpticalEditorApp",
    "launch_optical_editor",
]


def __getattr__(
    name,
):
    if name in __all__:
        from .editor import (
            OpticalArchitectureModel,
            OpticalEditorApp,
            launch_optical_editor,
        )

        return {
            "OpticalArchitectureModel": OpticalArchitectureModel,
            "OpticalEditorApp": OpticalEditorApp,
            "launch_optical_editor": launch_optical_editor,
        }[name]

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )

