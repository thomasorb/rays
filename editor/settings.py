from __future__ import annotations

import json

from pathlib import Path


# ==========================================================
# Paths
# ==========================================================

CONFIG_DIR = (
    Path.home()
    / ".config"
    / "rays"
)

CONFIG_FILE = (
    CONFIG_DIR
    / "editor.json"
)


# ==========================================================
# Utilities
# ==========================================================

def ensure_config_dir():

    CONFIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ==========================================================
# Load
# ==========================================================

def load_settings():

    ensure_config_dir()

    if not CONFIG_FILE.exists():

        return {}

    try:

        return json.loads(
            CONFIG_FILE.read_text()
        )

    except Exception:

        return {}


# ==========================================================
# Save
# ==========================================================

def save_settings(
    settings,
):

    ensure_config_dir()

    CONFIG_FILE.write_text(
        json.dumps(
            settings,
            indent=2,
        )
    )


# ==========================================================
# Convenience helpers
# ==========================================================

def get_setting(
    key,
    default=None,
):

    settings = load_settings()

    return settings.get(
        key,
        default,
    )


def set_setting(
    key,
    value,
):

    settings = load_settings()

    settings[key] = value

    save_settings(
        settings
    )

# ==========================================================
# File dialogs
# ==========================================================

def get_last_directory():

    directory = get_setting(
        "last_directory",
        None,
    )

    if directory:

        path = Path(directory)

        if path.exists():

            return str(path)

    return str(
        Path.home()
    )


def set_last_directory(
    path,
):

    path = Path(path)

    set_setting(
        "last_directory",
        str(path),
    )


def get_dialog_directory(
    current_file=None,
):
    """
    Preferred directory for open/save dialogs.

    Priority:

        1. current file directory
        2. last directory used
        3. home directory
    """

    if current_file:

        current_path = Path(
            current_file
        )

        if current_path.exists():

            return str(
                current_path.parent
            )

    return get_last_directory()
