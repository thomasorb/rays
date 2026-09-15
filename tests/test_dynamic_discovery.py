import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from registry import discover_material_specs
from registry import discover_optic_specs


def test_dynamic_material_discovery_includes_existing_materials():
    labels = {
        spec.label
        for spec in discover_material_specs()
    }

    assert "Air" in labels
    assert "BK7" in labels
    assert "Fused Silica" in labels


def test_dynamic_optic_discovery_includes_existing_optics():
    specs = {
        spec.name: spec
        for spec in discover_optic_specs()
    }

    assert "CornerCube" in specs
    assert "Detector" in specs
    assert "Mirror" in specs
    assert "PlateBeamSplitter" in specs
    assert "optical_axis" in {
        parameter.name
        for parameter in specs[
            "CornerCube"
        ].parameters
    }
