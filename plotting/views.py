from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Generic utilities
# ==========================================================

VIEW_MAP = {
    "xy": (0, 1),
    "xz": (0, 2),
    "yz": (1, 2),
}


def _get_axes(view):

    try:
        return VIEW_MAP[view]
    except KeyError:
        raise ValueError(
            f"Unknown view '{view}'"
        )


# ==========================================================
# Rays
# ==========================================================


def plot_ray(
    ray,
    view="xz",
    ax=None,
    **kwargs,
):

    if ax is None:
        fig, ax = plt.subplots()

    i, j = _get_axes(view)

    path = np.asarray(
        ray.path
    )

    ax.plot(
        path[:, i],
        path[:, j],
        "-",
        lw=1,
        color="crimson",
        alpha=0.1,
        zorder=1,
    )

    return ax

# ==========================================================
# Optical elements
# ==========================================================

def plot_element(
    element,
    view="xz",
    ax=None,
):
    """
    Pretty optical rendering.
    """

    if ax is None:
        fig, ax = plt.subplots()

    classname = (
        element.__class__.__name__
    )

    #
    # Style
    #

    if classname == "Mirror":

        color = "royalblue"

    elif classname == "BeamSplitter":

        color = "crimson"

    elif classname == "Compensator":

        color = "darkorange"

    elif classname == "Window":

        color = "purple"

    elif classname == "Detector":

        color = "forestgreen"

    else:

        color = "black"

    #
    # Volumes
    #

    if hasattr(
        element,
        "get_volume_vertices",
    ):

        plot_volume_element(
            element,
            view,
            ax,
            color,
        )

    #
    # Surfaces
    #
    elif hasattr(
        element,
        "get_outline",
    ):

        outline = (
            element.get_outline()
        )

        i, j = _get_axes(view)

        ax.plot(
            outline[:, i],
            outline[:, j],

            color=color,

            linewidth=3,

            zorder=20,
        )

   
    #
    # Label
    #

    center = (
        element.transform.position
    )

    i, j = _get_axes(view)

    ax.text(
        center[i],
        center[j],

        element.name,

        fontsize=8,

        ha="center",

        zorder=30,
    )

    #
    # Optional overlay
    #
    if hasattr(
        element,
        "draw_overlay",
    ):
        element.draw_overlay(
            view,
            ax,
        )

    return ax

def plot_volume_element(
    element,
    view,
    ax,
    color,
):
    """
    Draw a rectangular glass volume.
    """

    vertices = (
        element.get_volume_vertices()
    )

    i, j = _get_axes(view)

    #
    # Front / back faces
    #

    front = [0,1,2,3,0]
    back  = [4,5,6,7,4]

    ax.fill(
        vertices[front, i],
        vertices[front, j],

        color=color,

        alpha=0.15,

        zorder=5,
    )

    ax.fill(
        vertices[back, i],
        vertices[back, j],

        color=color,

        alpha=0.15,

        zorder=5,
    )

    ax.plot(
        vertices[front, i],
        vertices[front, j],

        color=color,

        linewidth=2,

        zorder=20,
    )

    ax.plot(
        vertices[back, i],
        vertices[back, j],

        color=color,

        linewidth=2,

        zorder=20,
    )

    #
    # Connecting edges
    #

    edges = [
        (0,4),
        (1,5),
        (2,6),
        (3,7),
    ]

    for a,b in edges:

        ax.plot(
            [
                vertices[a, i],
                vertices[b, i],
            ],
            [
                vertices[a, j],
                vertices[b, j],
            ],

            color=color,

            linewidth=1,

            alpha=0.7,

            zorder=10,
        )

def plot_system(
    tracer,
    rays=None,
    view="xz",
    figsize=(8,6),
    ax=None,
):
    """
    Complete system drawing.
    """

    if ax is None:

        fig, ax = plt.subplots(
            figsize=figsize
        )

    else:

        fig = ax.figure

    for element in tracer.elements:

        plot_element(
            element,
            view=view,
            ax=ax,
        )

    if rays is not None:

        if not isinstance(
            rays,
            (list, tuple),
        ):
            rays = [rays]

        for ray in rays:

            plot_ray(
                ray,
                view=view,
                ax=ax,
            )


    ax.grid(True)

    ax.set_aspect("equal")

    return fig, ax


# ==========================================================
# Monte-Carlo plots
# ==========================================================

def plot_detector_spots(
    xs,
    ys,
):
    """
    Spot diagram.
    """

    fig, ax = plt.subplots()

    ax.scatter(
        xs,
        ys,
        s=2,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax.axis("equal")
    ax.grid(True)

    return fig, ax


def plot_opl_histogram(
    opls,
):
    """
    Optical path histogram.
    """

    fig, ax = plt.subplots()

    ax.hist(
        opls,
        bins=50,
    )

    ax.set_xlabel(
        "Optical path length"
    )

    ax.set_ylabel(
        "Count"
    )

    return fig, ax


def plot_phase_histogram(
    phases,
):
    """
    Phase histogram.
    """

    fig, ax = plt.subplots()

    ax.hist(
        phases,
        bins=50,
    )

    ax.set_xlabel(
        "Phase [rad]"
    )

    ax.set_ylabel(
        "Count"
    )

    return fig, ax

def plot_system_3views(
    tracer,
    rays,
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15,5),
    )

    plot_system(
        tracer,
        rays,
        view="xy",
        ax=axes[0],
    )

    plot_system(
        tracer,
        rays,
        view="xz",
        ax=axes[1],
    )

    plot_system(
        tracer,
        rays,
        view="yz",
        ax=axes[2],
    )

    axes[0].set_title("XY")
    axes[1].set_title("XZ")
    axes[2].set_title("YZ")

    plt.tight_layout()

def plot_overlay(
    element,
    view="xz",
    ax=None,
):

    if hasattr(
        element,
        "draw_overlay",
    ):

        element.draw_overlay(
            view,
            ax,
        )
