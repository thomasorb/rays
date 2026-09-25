from tkinter import ttk


# ==========================================================
# Theme configuration
# ==========================================================

THEME_NAME = "darkly"

FONT_FAMILY = "Noto Sans"

FONT_SIZE = 9

APP_FONT = (
    FONT_FAMILY,
    FONT_SIZE,
)


def configure_theme():

    style = ttk.Style()

    #
    # Global default font
    #

    style.configure(
        ".",
        font=APP_FONT,
    )

    #
    # Entry widgets
    #
    # ttkbootstrap sometimes keeps
    # a larger default font for
    # these.
    #

    style.configure(
        "TEntry",
        font=APP_FONT,
    )

    #
    # Combo boxes
    #

    style.configure(
        "TCombobox",
        font=APP_FONT,
    )

    #
    # Notebook tabs
    #

    style.configure(
        "TNotebook.Tab",
        font=APP_FONT,
        padding=(8, 3),
    )

    #
    # Optional heading style
    #

    style.configure(
        "Heading.TLabel",
        font=(
            FONT_FAMILY,
            FONT_SIZE + 1,
            "bold",
        ),
    )


    return style
