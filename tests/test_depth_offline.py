import importlib.util
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest


def module():
    path = Path(__file__).parents[1] / "scripts/depth_anything_overlay.py"
    spec = importlib.util.spec_from_file_location("depth_overlay", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_local_only_loading_never_passes_a_remote_model_id(tmp_path, monkeypatch):
    script = module()
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"mock weights")
    calls = []

    class Loader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append((path, kwargs))
            return object()

    captured = {}

    def pipeline(**kwargs):
        captured.update(kwargs)
        return "fixture"

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoImageProcessor=Loader, AutoModelForDepthEstimation=Loader, pipeline=pipeline
        ),
    )
    # The helper sets these; restore the process environment after the test.
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    assert script.load_local_estimator(tmp_path, -1) == "fixture"
    assert len(calls) == 2
    assert all(
        path == str(tmp_path) and options == {"local_files_only": True, "trust_remote_code": False}
        for path, options in calls
    )
    assert not isinstance(captured["model"], str)
    assert not isinstance(captured["image_processor"], str)
    assert script.local_model_manifest(tmp_path)["model.safetensors"] == script.sha256_file(
        tmp_path / "model.safetensors"
    )


def test_remote_id_missing_weights_and_symlink_escape_rejected(tmp_path):
    script = module()
    with pytest.raises(ValueError, match="local"):
        script.local_model_manifest(Path("depth-anything/Depth-Anything-V2-Small-hf"))
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="weights"):
        script.local_model_manifest(model)
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside")
    (model / "weights.safetensors").symlink_to(outside)
    with pytest.raises(ValueError, match="inside"):
        script.local_model_manifest(model)


def test_experiment_preserves_raw_float_depth_and_labels_png_visual_only(tmp_path, monkeypatch):
    import json

    import numpy as np
    from PIL import Image

    script = module()
    model = tmp_path / "weights"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"mock weights")
    inputs = tmp_path / "images"
    inputs.mkdir()
    image_path = inputs / "frame.png"
    Image.new("RGB", (8, 6), color="gray").save(image_path)
    original = image_path.read_bytes()
    raw = np.array([[[0.125, 100.5, -2.25], [8.75, 0.75, 9.5]]], dtype=np.float32)

    class Tensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return raw

    class Estimator:
        device = "cpu"
        model = SimpleNamespace(config=SimpleNamespace(model_type="fixture_relative_depth"))

        def __call__(self, image):
            return {"predicted_depth": Tensor(), "depth": Image.new("L", image.size, 42)}

    monkeypatch.setattr(script, "load_local_estimator", lambda *_: Estimator())
    monkeypatch.setattr(script, "version", lambda _: "fixture")
    output = tmp_path / "outputs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "depth",
            "--input-dir",
            str(inputs),
            "--output-dir",
            str(output),
            "--model-path",
            str(model),
        ],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert script.main() == 0
    assert not [item for item in caught if issubclass(item.category, DeprecationWarning)]
    saved = np.load(output / "frame_relative_depth.npy", allow_pickle=False)
    assert saved.dtype == np.float32 and np.array_equal(saved, raw[0])
    evidence = json.loads((output / "depth_anything_evidence.json").read_text())
    assert evidence["measurement_allowed"] is False
    assert evidence["depth_semantics"] == "RELATIVE_NOT_METRIC"
    assert evidence["model"]["executed_model_type"] == "fixture_relative_depth"
    assert evidence["samples"][0]["depth_png_semantics"] == "NORMALIZED_8BIT_VISUALIZATION_ONLY"
    assert evidence["samples"][0]["raw_shape"] == [2, 3]
    assert image_path.read_bytes() == original
