from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from plotting.editor import (
    ComponentSpec,
    OpticalArchitectureModel,
    SourceSpec,
)


class OpticalEditorModelTests(
    unittest.TestCase
):
    def test_round_trip_json_preserves_editor_state(
        self,
    ):
        model = OpticalArchitectureModel(
            components=[
                ComponentSpec(
                    component_type="PlateBeamSplitter",
                    name="BSX",
                    position=[1.0, 2.0, 3.0],
                    rotation_deg=[0.0, 45.0, 0.0],
                    width=30.0,
                    height=20.0,
                    thickness=4.0,
                    material="Custom",
                    material_n=1.37,
                    R=0.4,
                    T=0.5,
                    back_R=0.1,
                    back_T=0.8,
                ),
                ComponentSpec(
                    component_type="Window",
                    name="WINX",
                    position=[10.0, -5.0, 2.0],
                    rotation_deg=[5.0, 0.0, 15.0],
                    width=12.0,
                    height=14.0,
                    thickness=3.0,
                    material="FusedSilica",
                ),
            ],
            source=SourceSpec(
                kind="bundle",
                position=[-20.0, 1.0, 0.5],
                direction=[1.0, 0.1, 0.0],
                wavelength=532e-9,
                radius=1.25,
                n_rays=17,
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "editor.json"
            model.save_json(path)
            loaded = OpticalArchitectureModel.load_json(
                path
            )

        self.assertEqual(
            loaded.to_dict(),
            model.to_dict(),
        )

    def test_loaded_model_builds_system(
        self,
    ):
        model = OpticalArchitectureModel(
            components=[
                ComponentSpec(
                    component_type="Mirror",
                    name="M10",
                    position=[0.0, 0.0, 0.0],
                    rotation_deg=[0.0, 0.0, 0.0],
                ),
                ComponentSpec(
                    component_type="Detector",
                    name="DET10",
                    position=[20.0, 0.0, 0.0],
                    rotation_deg=[0.0, -90.0, 0.0],
                ),
                ComponentSpec(
                    component_type="OpticalInterface",
                    name="IF10",
                    position=[5.0, 0.0, 0.0],
                    rotation_deg=[0.0, 90.0, 0.0],
                    material1="Air",
                    material1_n=1.0,
                    material2="Custom",
                    material2_n=1.42,
                    R=0.2,
                    T=0.7,
                ),
            ],
            source=SourceSpec(
                kind="single",
                position=[-10.0, 0.0, 0.0],
                direction=[1.0, 0.0, 0.0],
            ),
        )

        system = model.build_system()
        traced = system.trace()

        self.assertGreater(
            len(traced),
            0,
        )
        self.assertIsNotNone(
            system.detector_by_name("DET10")
        )


if __name__ == "__main__":
    unittest.main()

