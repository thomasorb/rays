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
    """
    Plot one ray.
    """

    if ax is None:
        fig, ax = plt.subplots()

    i, j = _get_axes(view)

    path = np.asarray(ray.path)

    ax.plot(
        path[:, i],
        path[:, j],
        "-o",
        **kwargs,
    )

    ax.set_aspect("equal")

    return ax


def plot_rays(
    rays,
    view="xz",
    ax=None,
):
    """
    Plot multiple rays.
    """

    if ax is None:
        fig, ax = plt.subplots()

    for ray in rays:
        plot_ray(
            ray,
            view=view,
            ax=ax,
        )

    return ax


# ==========================================================
# Optical elements
# ==========================================================

def plot_element(
    element,
    view="xz",
    ax=None,
    color="black",
):
    """
    Plot an optical element.

    Requires get_outline().
    """

    if not hasattr(
        element,
        "get_outline"
    ):
        return ax

    if ax is None:
        fig, ax = plt.subplots()

    outline = element.get_outline()

    if outline is None:
        return ax

    i, j = _get_axes(view)

    ax.plot(
        outline[:, i],
        outline[:, j],
        color=color,
    )

    return ax


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
