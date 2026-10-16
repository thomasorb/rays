from __future__ import annotations

import inspect

LOGGER = None

DEBUG = True


# ==================================================
# Registration
# ==================================================

def set_logger(
    callback,
):
    global LOGGER

    LOGGER = callback


# ==================================================
# Helpers
# ==================================================

def _calling_module():

    stack = inspect.stack()

    #
    # stack[0]
    #   _calling_module
    #
    # stack[1]
    #   log
    #
    # stack[2]
    #   caller
    #

    if len(stack) < 3:
        return "unknown"

    frame = stack[2].frame

    module = inspect.getmodule(
        frame
    )

    if module is None:
        return "unknown"

    return module.__name__


# ==================================================
# Logging
# ==================================================

def log(
    *args,
):

    if not DEBUG:
        return

    message = " ".join(
        str(x)
        for x in args
    )

    prefix = (
        _calling_module()
    )

    formatted = (
        f"[{prefix}] {message}"
    )

    #
    # Editor logger
    #

    if LOGGER is not None:

        try:

            LOGGER(
                formatted
            )

        except Exception:

            print(
                formatted
            )

    #
    # Fallback console
    #

    else:

        print(
            formatted
        )
