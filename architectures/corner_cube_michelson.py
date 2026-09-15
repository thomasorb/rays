from __future__ import annotations

import numpy as np

from scipy.spatial.transform import Rotation

from system.optical_system import OpticalSystem

from optics.corner_cube import CornerCube
from optics.compensator import Compensator
from optics.plate_beamsplitter import PlateBeamSplitter
from optics.detector import Detector


class CornerCubeMichelson:
    """
    Michelson interferometer using
    two corner cubes.

    The horizontal corner cube is
    the scanning element.
    """

    def __init__(

        self,

        arm_length,

        compensator_distance,

        cube_size,

        plate_size,

        plate_thickness,

        material,

        detector_distance=50,
    ):

        self.arm_length = float(
            arm_length
        )

        self.compensator_distance = float(
            compensator_distance
        )

        self.detector_distance = float(
            detector_distance
        )

        self.system = OpticalSystem()

        # ==================================================
        # BEAMSPLITTER
        # ==================================================

        self.bs = PlateBeamSplitter(

            name="BS",

            position=[0,0,0],

            rotation=Rotation.from_euler(
                "y",
                135,
                degrees=True,
            ),

            thickness=plate_thickness,

            width=plate_size,
            height=plate_size,

            material=material,

            R=0.5,
            T=0.5,
        )

        self.bs.add_to_system(
            self.system
        )

        # ==================================================
        # COMPENSATOR
        # ==================================================

        self.compensator = Compensator(

            name="COMP",

            position=[
                0,
                0,
                compensator_distance,
            ],

            rotation=Rotation.from_euler(
                "y",
                135,
                degrees=True,
            ),

            thickness=plate_thickness,

            width=plate_size,
            height=plate_size,

            material=material,
        )

        self.compensator.add_to_system(
            self.system
        )

        # ==================================================
        # FIXED CORNER CUBE
        # ==================================================

        self.corner_cube1 = CornerCube(

            name="CC1",

            position=[
                0,
                0,
                arm_length,
            ],

            size=cube_size,
        )

        self.corner_cube1.add_to_system(
            self.system
        )

        # ==================================================
        # MOVING CORNER CUBE
        # ==================================================

        self.corner_cube2 = CornerCube(

            name="CC2",

            position=[
                arm_length,
                0,
                0,
            ],

            size=cube_size,
        )

        self.corner_cube2.add_to_system(
            self.system
        )

        # ==================================================
        # DETECTOR 1
        # ==================================================

        self.detector1 = Detector(

            name="DET1",

            position=[
                -detector_distance,
                0,
                0,
            ],

            rotation=Rotation.from_euler(
                "y",
                -90,
                degrees=True,
            ),

            width=plate_size,
            height=plate_size,
        )

        self.system.add_detector(
            self.detector1
        )

        # ==================================================
        # DETECTOR 2
        # ==================================================

        self.detector2 = Detector(

            name="DET2",

            position=[
                0,
                0,
                -detector_distance,
            ],

            rotation=Rotation.identity(),

            width=plate_size,
            height=plate_size,
        )

        self.system.add_detector(
            self.detector2
        )

        # ==================================================
        # STORE ZPD
        # ==================================================

        self.zpd_position = (
            self.corner_cube2.position.copy()
        )

    # ======================================================
    # SOURCE
    # ======================================================

    def set_source(
        self,
        source,
    ):

        self.source = source

        self.system.set_source(
            source
        )

    # ======================================================
    # TRACE
    # ======================================================

    def trace(
        self,
        progress_callback=None,
    ):

        return self.system.trace(
            progress_callback=
                progress_callback
        )

    # ======================================================
    # MIRROR MOTION
    # ======================================================

    def move_mirror(
        self,
        displacement,
    ):
        """
        Move the horizontal corner cube.

        OPD = 2 × displacement.
        """

        new_position = (
            self.zpd_position
            +
            np.array(
                [
                    displacement,
                    0,
                    0,
                ]
            )
        )

        self.corner_cube2.position = (
            new_position
        )

    # ------------------------------------------------------

    def reset_zpd(
        self,
    ):

        self.corner_cube2.position = (
            self.zpd_position.copy()
        )

    # ======================================================
    # SCAN
    # ======================================================

    def scan_mirror(
        self,
        mirror_positions,
        progress_callback=None,
    ):

        mirror_positions = np.asarray(
            mirror_positions,
            dtype=float,
        )

        signal_det1 = []
        signal_det2 = []

        try:

            n_positions = len(
                mirror_positions
            )

            for index, dx in enumerate(
                mirror_positions,
                start=1,
            ):

                def step_progress(
                    value=None,
                    message=None,
                    current=None,
                    total=None,
                ):
                    if progress_callback is None:
                        return

                    if value is None:
                        progress_callback(
                            value=None,
                            message=(
                                message
                                or
                                f"Tracing move step "
                                f"{index}/{n_positions}"
                            ),
                            current=index,
                            total=n_positions,
                        )
                        return

                    progress_callback(
                        value=(
                            (
                                index - 1
                                + value
                            )
                            / n_positions
                        ),
                        message=(
                            message
                            or
                            f"Tracing move step "
                            f"{index}/{n_positions}"
                        ),
                        current=index,
                        total=n_positions,
                    )

                self.move_mirror(
                    dx
                )

                if progress_callback is not None:
                    progress_callback(
                        value=(
                            (index - 1)
                            / n_positions
                        ),
                        message=(
                            f"Moving mirror "
                            f"{index}/{n_positions}"
                        ),
                        current=index - 1,
                        total=n_positions,
                    )

                self.trace(
                    progress_callback=
                        step_progress
                )

                signal_det1.append(
                    self.detector1.intensity
                )

                signal_det2.append(
                    self.detector2.intensity
                )

                if progress_callback is not None:
                    progress_callback(
                        value=index / n_positions,
                        message=(
                            f"Completed move step "
                            f"{index}/{n_positions}"
                        ),
                        current=index,
                        total=n_positions,
                    )

        finally:

            self.reset_zpd()

        return {

            "mirror_positions":
                mirror_positions,

            "opd":
                2.0 * mirror_positions,

            "det1":
                np.asarray(signal_det1),

            "det2":
                np.asarray(signal_det2),
        }

    # ======================================================
    # REPORT
    # ======================================================

    def report(
        self,
    ):

        print()
        print("================================")
        print("CORNER CUBE MICHELSON")
        print("================================")

        print()

        print(
            "Arm length:",
            self.arm_length
        )

        print(
            "Compensator distance:",
            self.compensator_distance
        )

        print(
            "CC2 position:",
            self.corner_cube2.position
        )
