import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from sih26158.process_control import (
    ManagedProcessExecutor,
    ProcessCancelledError,
    ProcessTimeoutError,
)
from sih26158.segmentation import (
    SegmentationSettings,
    run_managed_segmentation,
    run_optional_segmentation,
)


def test_mocked_segmentation_declares_masks_without_model_download(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    image_path = frames / "frame_000001.jpg"
    cv2.imwrite(str(image_path), np.zeros((20, 30, 3), np.uint8))
    keyframes: list[dict[str, object]] = [
        {"frame_index": 1, "image_name": image_path.name, "selected": True}
    ]

    def provider(_: Path) -> np.ndarray:
        mask = np.zeros((20, 30), np.uint8)
        mask[:, :3] = 255
        return mask

    artifacts, warnings = run_optional_segmentation(
        tmp_path, "run_test", keyframes, None, provider=provider
    )
    assert not warnings
    assert {path.name for path in artifacts} == {
        "frame_000001_dynamic.png",
        "frame_000001.jpg.png",
        "frame_000001.jpg.mask.png",
        "segmentation_comparison.json",
        "segmentation_contact_sheet.jpg",
        "segmentation_review.json",
    }
    assert keyframes[0]["dynamic_mask_fraction"] == 0.1
    assert str(keyframes[0]["mask_url"]).startswith("/api/runs/run_test/artifacts/masks/")


def test_missing_optional_model_falls_back_to_unmasked_frames(tmp_path: Path) -> None:
    artifacts, warnings = run_optional_segmentation(
        tmp_path, "run_test", [], str(tmp_path / "missing.pt")
    )
    assert [path.name for path in artifacts] == ["segmentation_comparison.json"]
    assert warnings[0]["code"] == "SEGMENTATION_UNAVAILABLE_USING_UNMASKED_FRAMES"


def test_primary_subject_missing_model_is_a_blocker(tmp_path: Path) -> None:
    artifacts, warnings = run_optional_segmentation(
        tmp_path,
        "run_test",
        [],
        str(tmp_path / "missing.pt"),
        reconstruction_target="PRIMARY_SUBJECT",
        masking_mode="AUTO",
    )
    report = json.loads(artifacts[0].read_text())
    assert report["status"] == "BLOCKED"
    assert warnings[0]["code"] == "REQUIRED_SEGMENTATION_UNAVAILABLE"


def test_operational_masks_invert_review_exclusion_semantics(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    image_path = frames / "frame.jpg"
    cv2.imwrite(str(image_path), np.zeros((10, 10, 3), np.uint8))
    keyframes: list[dict[str, object]] = [
        {"frame_index": 0, "image_name": image_path.name, "selected": True}
    ]

    def provider(_: Path) -> np.ndarray:
        mask = np.zeros((10, 10), np.uint8)
        mask[:, :2] = 255
        return mask

    run_optional_segmentation(tmp_path, "run_test", keyframes, None, provider=provider)
    review = cv2.imread(str(tmp_path / "masks/frame_dynamic.png"), cv2.IMREAD_GRAYSCALE)
    operational = cv2.imread(
        str(tmp_path / "masks/reconstruction/frame.jpg.mask.png"),
        cv2.IMREAD_GRAYSCALE,
    )
    assert np.all(review[:, :2] == 255)
    assert np.all(operational[:, :2] == 0)
    assert np.all(operational[:, 2:] == 255)


def test_primary_subject_rejects_trivial_keep_everything_mask(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    image_path = frames / "frame.jpg"
    cv2.imwrite(str(image_path), np.zeros((10, 10, 3), np.uint8))
    keyframes: list[dict[str, object]] = [
        {"frame_index": 0, "image_name": image_path.name, "selected": True}
    ]

    artifacts, warnings = run_optional_segmentation(
        tmp_path,
        "run_test",
        keyframes,
        None,
        provider=lambda _: np.zeros((10, 10), np.uint8),
        reconstruction_target="PRIMARY_SUBJECT",
        masking_mode="AUTO",
    )

    assert [path.name for path in artifacts] == ["segmentation_comparison.json"]
    assert warnings[0]["code"] == "REQUIRED_SEGMENTATION_FAILED"


def _frames(root: Path, count: int = 1) -> list[dict]:
    (root / "frames").mkdir(exist_ok=True)
    frames = []
    for i in range(count):
        name = f"frame_{i}.png"
        cv2.imwrite(str(root / "frames" / name), np.full((12, 20, 3), 120, np.uint8))
        frames.append({"image_name": name, "selected": True})
    return frames


def _report(root: Path) -> dict:
    return json.loads((root / "segmentation_comparison.json").read_text())


def test_environment_weights_resolve_provider_hash_and_cache_dependency(tmp_path, monkeypatch):
    from sih26158 import segmentation
    from sih26158.storage import sha256_file

    weights = tmp_path / "local.pt"
    weights.write_bytes(b"fixture checkpoint; mocked loader")
    monkeypatch.setenv("SIH_SEGMENTATION_MODEL", str(weights))

    def loader(path, target, settings):
        assert path == weights and target == "FULL_SCENE" and settings.device == "cpu"

        def provider(_):
            return np.zeros((12, 20), np.uint8)

        provider.execution_metadata = {"actual_device": "cpu", "supported_classes": ["car"]}
        return provider

    monkeypatch.setattr(segmentation, "_ultralytics_provider", loader)
    frames = _frames(tmp_path)
    segmentation.run_optional_segmentation(tmp_path, "r", frames, None)
    report = _report(tmp_path)
    assert report["provider"] == "YOLO_SEGMENTATION_LOCAL_WEIGHTS"
    assert report["execution"]["model_source"] == "ENVIRONMENT"
    assert report["execution"]["model_sha256"] == sha256_file(weights)
    assert report["execution"]["actual_device"] == "cpu"
    assert report["runtime_s"] >= 0
    old = segmentation.segmentation_fingerprint_inputs(None)
    weights.write_bytes(b"changed weights")
    assert old != segmentation.segmentation_fingerprint_inputs(None)
    changed = segmentation.SegmentationSettings(confidence=0.8)
    assert segmentation.segmentation_fingerprint_inputs(
        None, changed
    ) != segmentation.segmentation_fingerprint_inputs(None)


def test_no_detections_is_valid_unmasked_evidence_not_primary_subject(tmp_path):
    frames = _frames(tmp_path)
    run_optional_segmentation(
        tmp_path, "r", frames, "unused.pt", provider=lambda _: np.zeros((12, 20))
    )
    report = _report(tmp_path)
    assert report["status"] == "APPLIED" and report["mean_excluded_fraction"] == 0
    assert report["provider"] == "INJECTED_PROVIDER"
    assert report["execution"]["model_source"] == "INJECTED"
    assert report["execution"]["actual_device"] is None
    review = json.loads((tmp_path / "segmentation_review.json").read_text())
    assert review["status"] == "AWAITING_MANUAL_REVIEW"
    assert review["samples"][0]["missed_objects"] is None
    assert cv2.imread(str(tmp_path / "segmentation_contact_sheet.jpg")) is not None


def test_partial_provider_failure_cleans_all_files_and_metadata(tmp_path):
    frames = _frames(tmp_path, 2)
    calls = 0

    def provider(_):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("fixture failure")
        return np.zeros((12, 20), np.uint8)

    artifacts, warnings = run_optional_segmentation(
        tmp_path, "r", frames, None, provider=provider, masking_mode="REQUIRED"
    )
    assert _report(tmp_path)["status"] == "BLOCKED"
    assert warnings[0]["code"] == "REQUIRED_SEGMENTATION_FAILED"
    assert len(artifacts) == 1
    assert not list((tmp_path / "masks").rglob("*.png"))
    assert all("mask_url" not in frame for frame in frames)


def test_invalid_local_checkpoint_reports_unavailable_without_download(tmp_path, monkeypatch):
    from sih26158 import segmentation

    weights = tmp_path / "bad.pt"
    weights.write_bytes(b"invalid")

    def invalid(*args):
        raise ValueError("Invalid local checkpoint")

    monkeypatch.setattr(segmentation, "_ultralytics_provider", invalid)
    run_optional_segmentation(tmp_path, "r", _frames(tmp_path), str(weights))
    assert _report(tmp_path)["status"] == "UNAVAILABLE_FALLBACK"
    assert _report(tmp_path)["execution"]["executed_provider"] is None


def test_nonzero_polarity_and_nearest_resize_preserve_source_bytes(tmp_path):
    frames = _frames(tmp_path)
    source = tmp_path / "frames" / frames[0]["image_name"]
    original = source.read_bytes()
    mask = np.zeros((6, 10), float)
    mask[:, 0] = -0.2
    run_optional_segmentation(tmp_path, "r", frames, None, provider=lambda _, mask=mask: mask)
    review = cv2.imread(str(tmp_path / "masks/frame_0_dynamic.png"), 0)
    assert np.all(review[:, :2] == 255) and np.all(review[:, 2:] == 0)
    assert source.read_bytes() == original
    for name in ("frame_0.png.png", "frame_0.png.mask.png"):
        operational = cv2.imread(str(tmp_path / "masks/reconstruction" / name), 0)
        assert np.array_equal(operational, 255 - review)


def test_invalid_masks_and_near_total_masks_fail_safely(tmp_path):
    frames = _frames(tmp_path)
    for mask in (np.ones((12, 20)), np.full((12, 20), np.nan), np.zeros((12, 20, 3))):
        run_optional_segmentation(tmp_path, "r", frames, None, provider=lambda _, mask=mask: mask)
        assert _report(tmp_path)["status"] == "FAILED_FALLBACK"
        assert not list((tmp_path / "masks").rglob("*.png"))


def test_off_does_not_load_weights_and_cleans_previous_masks(tmp_path, monkeypatch):
    from sih26158 import segmentation

    frames = _frames(tmp_path)
    run_optional_segmentation(tmp_path, "r", frames, None, provider=lambda _: np.zeros((12, 20)))

    def unexpected(*args):
        raise AssertionError("OFF must not load a model")

    monkeypatch.setattr(segmentation, "_ultralytics_provider", unexpected)
    run_optional_segmentation(tmp_path, "r", frames, None, masking_mode="OFF")
    assert _report(tmp_path)["status"] == "DISABLED"
    assert not list((tmp_path / "masks").rglob("*.png"))
    assert not (tmp_path / "segmentation_contact_sheet.jpg").exists()


def test_cancellation_timeout_limits_and_accelerator_allocation(tmp_path):
    from sih26158.process_control import ProcessCancelledError, ProcessTimeoutError
    from sih26158.segmentation import SegmentationSettings

    frames = _frames(tmp_path, 2)
    provider = lambda _: np.zeros((12, 20))
    with pytest.raises(ProcessCancelledError):
        run_optional_segmentation(
            tmp_path, "r", frames, None, provider=provider, cancel_requested=lambda: True
        )
    with pytest.raises(ProcessTimeoutError):
        run_optional_segmentation(
            tmp_path,
            "r",
            frames,
            None,
            provider=provider,
            settings=SegmentationSettings(timeout_s=1e-12),
        )
    with pytest.raises(ValueError, match="scheduler"):
        run_optional_segmentation(
            tmp_path,
            "r",
            frames,
            None,
            provider=provider,
            settings=SegmentationSettings(device="cuda:0"),
        )
    run_optional_segmentation(
        tmp_path, "r", frames, None, provider=provider, settings=SegmentationSettings(max_frames=1)
    )
    assert _report(tmp_path)["status"] == "FAILED_FALLBACK"


def test_class_policy_uses_actual_model_names_and_explicit_predict_settings(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from sih26158.segmentation import SegmentationSettings, _ultralytics_provider

    frames = _frames(tmp_path)
    path = tmp_path / "local.pt"
    path.write_bytes(b"mocked")
    calls = []

    class Tensor:
        def cpu(self):
            return self

        def numpy(self):
            # One person, dog and sky, distinct columns.
            masks = np.zeros((3, 12, 20))
            for i in range(3):
                masks[i, :, i] = 1
            return masks

    class YOLO:
        task = "segment"
        predictor = SimpleNamespace(device="cpu")

        def __init__(self, value):
            self.names = {0: "person", 1: "dog", 2: "sky"}
            assert value == str(path)

        def predict(self, **kwargs):
            calls.append(kwargs)
            return [
                SimpleNamespace(
                    masks=SimpleNamespace(data=Tensor()),
                    boxes=SimpleNamespace(cls=np.array([0, 1, 2])),
                    names=self.names,
                )
            ]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=YOLO))
    provider = _ultralytics_provider(path, "FULL_SCENE", SegmentationSettings())
    mask = provider(tmp_path / "frames" / frames[0]["image_name"])
    assert np.all(mask[:, 0] == 255) and np.all(mask[:, 1] == 0) and np.all(mask[:, 2] == 255)
    assert calls[0]["device"] == "cpu" and calls[0]["retina_masks"] is True
    assert calls[0]["max_det"] == 100 and calls[0]["imgsz"] == 640
    assert calls[0]["iou"] == 0.7


@pytest.mark.parametrize("cancel", [False, True])
def test_managed_segmentation_kills_blocked_worker_and_cleans_partial_outputs(
    tmp_path: Path, cancel: bool
) -> None:
    frames = _frames(tmp_path)
    (tmp_path / "keyframes.json").write_text(json.dumps({"frames": frames}))
    partial = tmp_path / "masks" / "frame_0_dynamic.png"
    script = (
        "from pathlib import Path; import time; "
        f"p=Path({str(partial)!r}); p.parent.mkdir(parents=True, exist_ok=True); "
        "p.write_bytes(b'partial'); time.sleep(30)"
    )
    started = time.monotonic()
    executor = ManagedProcessExecutor(
        timeout_s=5 if cancel else 0.2,
        cancel_requested=(lambda: time.monotonic() - started > 0.1) if cancel else (lambda: False),
        heartbeat=lambda: None,
        poll_interval_s=0.02,
        terminate_grace_s=0.1,
    )
    expected = ProcessCancelledError if cancel else ProcessTimeoutError
    with pytest.raises(expected):
        run_managed_segmentation(
            tmp_path,
            "run_test",
            frames,
            None,
            reconstruction_target="FULL_SCENE",
            masking_mode="AUTO",
            settings=SegmentationSettings(timeout_s=5),
            executor=executor,
            worker_command=[sys.executable, "-c", script],
        )
    assert time.monotonic() - started < 3
    assert not partial.exists()
    assert not (tmp_path / ".segmentation-worker-outcome.json").exists()
