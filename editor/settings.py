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

