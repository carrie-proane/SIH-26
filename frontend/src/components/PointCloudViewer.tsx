import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";

import { resolveAssetUrl } from "../api";
import type {
  CameraPose,
  ConfidenceLabel,
  MeasurementResult,
  ViewerManifest,
} from "../types";

interface PointCloudViewerProps {
  manifest: ViewerManifest;
  cameraPoses: CameraPose[];
  visibleLabels: Set<ConfidenceLabel>;
  measurementEnabled: boolean;
  measurementResetKey: number;
  selectedFrameIndex: number | null;
  onMeasurementChange: (result: MeasurementResult) => void;
}

interface PickedPoint {
  position: THREE.Vector3;
  label: ConfidenceLabel;
}

const DISABLED_LABELS = new Set<ConfidenceLabel>([
  "AI_ASSISTED_NOT_MEASURABLE",
  "UNSEEN",
]);

function scenePosition(x: number, y: number, z: number): THREE.Vector3 {
  return new THREE.Vector3(x, z, -y);
}

function hexToRgb(hex: string): [number, number, number] {
  const value = Number.parseInt(hex.replace("#", ""), 16);
  return [((value >> 16) & 255) / 255, ((value >> 8) & 255) / 255, (value & 255) / 255];
}

function nearestConfidence(
  red: number,
  green: number,
  blue: number,
  manifest: ViewerManifest,
): ConfidenceLabel {
  let nearest = manifest.confidence_legend[0]?.label ?? "OBSERVED_MEDIUM";
  let nearestDistance = Number.POSITIVE_INFINITY;
  for (const item of manifest.confidence_legend) {
    const [targetRed, targetGreen, targetBlue] = hexToRgb(item.color);
    const distance =
      (red - targetRed) ** 2 + (green - targetGreen) ** 2 + (blue - targetBlue) ** 2;
    if (distance < nearestDistance) {
      nearestDistance = distance;
      nearest = item.label;
    }
  }
  return nearest;
}

function makeMarker(position: THREE.Vector3, color: string, radius = 0.075): THREE.Mesh {
  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 16, 12),
    new THREE.MeshBasicMaterial({ color, depthTest: false }),
  );
  marker.position.copy(position);
  marker.renderOrder = 10;
  return marker;
}

export function PointCloudViewer({
  manifest,
  cameraPoses,
  visibleLabels,
  measurementEnabled,
  measurementResetKey,
  selectedFrameIndex,
  onMeasurementChange,
}: PointCloudViewerProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const labelObjectsRef = useRef<Map<ConfidenceLabel, THREE.Points>>(new Map());
  const measurementGroupRef = useRef<THREE.Group | null>(null);
  const cameraMarkerGroupRef = useRef<THREE.Group | null>(null);
  const selectedCameraMarkerRef = useRef<THREE.Mesh | null>(null);
  const pickedPointsRef = useRef<PickedPoint[]>([]);
  const measurementEnabledRef = useRef(measurementEnabled);
  const onMeasurementChangeRef = useRef(onMeasurementChange);
  const visibleLabelsRef = useRef(visibleLabels);
  const [loadState, setLoadState] = useState<"LOADING" | "READY" | "ERROR">("LOADING");
  const [loadError, setLoadError] = useState("");

  measurementEnabledRef.current = measurementEnabled;
  onMeasurementChangeRef.current = onMeasurementChange;
  visibleLabelsRef.current = visibleLabels;

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    let disposed = false;
    let animationFrame = 0;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#07100f");
    scene.fog = new THREE.FogExp2("#07100f", 0.035);

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 5000);
    camera.position.set(7, 6, 8);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    host.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 0.25;
    controls.maxDistance = 500;

    const ambient = new THREE.AmbientLight("#d7fff6", 1.2);
    scene.add(ambient);
    const grid = new THREE.GridHelper(40, 40, "#1e6c5c", "#12342f");
    grid.position.y = -0.02;
    scene.add(grid);
    scene.add(new THREE.AxesHelper(0.9));

    const measurementGroup = new THREE.Group();
    measurementGroup.name = "measurements";
    scene.add(measurementGroup);
    measurementGroupRef.current = measurementGroup;
    const cameraMarkerGroup = new THREE.Group();
    cameraMarkerGroup.name = "selected-camera";
    scene.add(cameraMarkerGroup);
    cameraMarkerGroupRef.current = cameraMarkerGroup;

    const cameraPathPoints = cameraPoses.map((pose) => scenePosition(pose.x, pose.y, pose.z));
    if (cameraPathPoints.length > 1) {
      const path = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(cameraPathPoints),
        new THREE.LineBasicMaterial({ color: "#71f6d3", transparent: true, opacity: 0.86 }),
      );
      path.name = "camera-path";
      scene.add(path);
      const pathMarkers = new THREE.Points(
        new THREE.BufferGeometry().setFromPoints(cameraPathPoints),
        new THREE.PointsMaterial({ color: "#c5fff0", size: 0.075, sizeAttenuation: true }),
      );
      scene.add(pathMarkers);
    }

    const resize = () => {
      const width = Math.max(host.clientWidth, 1);
      const height = Math.max(host.clientHeight, 1);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(host);
    resize();

    const raycaster = new THREE.Raycaster();
    raycaster.params.Points = { threshold: 0.16 };
    const pointer = new THREE.Vector2();

    const clearMeasurement = () => {
      pickedPointsRef.current = [];
      measurementGroup.clear();
    };

    const onPointerDown = (event: PointerEvent) => {
      if (!measurementEnabledRef.current) return;
      const bounds = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
      pointer.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const objects = [...labelObjectsRef.current.values()].filter((item) => item.visible);
      const intersection = raycaster.intersectObjects(objects, false)[0];
      if (!intersection?.point) return;
      const label = intersection.object.userData.confidence as ConfidenceLabel;
      if (DISABLED_LABELS.has(label)) {
        onMeasurementChangeRef.current({
          distanceM: null,
          labels: [label],
          status: "BLOCKED",
          message: `${label} geometry is not eligible for verified measurement.`,
        });
        return;
      }

      if (pickedPointsRef.current.length >= 2) clearMeasurement();
      const picked = { position: intersection.point.clone(), label };
      pickedPointsRef.current.push(picked);
      measurementGroup.add(makeMarker(picked.position, "#f8fff9"));

      if (pickedPointsRef.current.length === 1) {
        onMeasurementChangeRef.current({
          distanceM: null,
          labels: [label],
          status: "SELECTING",
          message: "First observed point selected. Choose a second point.",
        });
        return;
      }

      const [first, second] = pickedPointsRef.current;
      const distance = first.position.distanceTo(second.position);
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([first.position, second.position]),
        new THREE.LineBasicMaterial({ color: "#f8fff9" }),
      );
      measurementGroup.add(line);
      const labels = [first.label, second.label];
      const status = labels.includes("OBSERVED_LOW")
        ? "CONFIRM"
        : labels.includes("OBSERVED_MEDIUM")
          ? "CAUTION"
          : "ALLOWED";
      onMeasurementChangeRef.current({
        distanceM: distance,
        labels,
        status,
        message:
          status === "ALLOWED"
            ? "Both points are supported by high-confidence observed geometry."
            : status === "CONFIRM"
              ? "Low-confidence geometry requires explicit operator confirmation."
              : "Measurement includes medium-confidence geometry; use with caution.",
      });
    };
    renderer.domElement.addEventListener("pointerdown", onPointerDown);

    const loader = new PLYLoader();
    loader.load(
      resolveAssetUrl(manifest.cloud.url),
      (geometry) => {
        if (disposed) {
          geometry.dispose();
          return;
        }
        const position = geometry.getAttribute("position");
        const color = geometry.getAttribute("color");
        const grouped = new Map<ConfidenceLabel, number[]>();
        for (const item of manifest.confidence_legend) grouped.set(item.label, []);
        for (let index = 0; index < position.count; index += 1) {
          const label = color
            ? nearestConfidence(color.getX(index), color.getY(index), color.getZ(index), manifest)
            : "OBSERVED_MEDIUM";
          grouped.get(label)?.push(
            position.getX(index),
            position.getZ(index),
            -position.getY(index),
          );
        }

        const pointGroup = new THREE.Group();
        pointGroup.name = "confidence-point-cloud";
        for (const item of manifest.confidence_legend) {
          const values = grouped.get(item.label) ?? [];
          if (!values.length) continue;
          const labelGeometry = new THREE.BufferGeometry();
          labelGeometry.setAttribute("position", new THREE.Float32BufferAttribute(values, 3));
          const points = new THREE.Points(
            labelGeometry,
            new THREE.PointsMaterial({
              color: item.color,
              size: 0.13,
              sizeAttenuation: true,
              transparent: true,
              opacity: 0.95,
            }),
          );
          points.userData.confidence = item.label;
          points.visible = visibleLabelsRef.current.has(item.label);
          labelObjectsRef.current.set(item.label, points);
          pointGroup.add(points);
        }
        scene.add(pointGroup);

        const bounds = new THREE.Box3().setFromObject(pointGroup);
        for (const point of cameraPathPoints) bounds.expandByPoint(point);
        if (!bounds.isEmpty()) {
          const center = bounds.getCenter(new THREE.Vector3());
          const size = Math.max(bounds.getSize(new THREE.Vector3()).length(), 1);
          controls.target.copy(center);
          camera.position.copy(center).add(new THREE.Vector3(size * 0.9, size * 0.65, size));
          camera.near = Math.max(size / 1000, 0.01);
          camera.far = size * 100;
          camera.updateProjectionMatrix();
          controls.update();
        }
        geometry.dispose();
        setLoadState("READY");
      },
      undefined,
      (error) => {
        if (disposed) return;
        setLoadError(error instanceof Error ? error.message : "Point cloud could not be loaded.");
        setLoadState("ERROR");
      },
    );

    const animate = () => {
      controls.update();
      renderer.render(scene, camera);
      animationFrame = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      disposed = true;
      cancelAnimationFrame(animationFrame);
      resizeObserver.disconnect();
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      controls.dispose();
      scene.traverse((object) => {
        if (object instanceof THREE.Points || object instanceof THREE.Line || object instanceof THREE.Mesh) {
          object.geometry?.dispose();
          const materials = Array.isArray(object.material) ? object.material : [object.material];
          materials.forEach((material) => material?.dispose());
        }
      });
      renderer.dispose();
      renderer.domElement.remove();
      labelObjectsRef.current.clear();
      measurementGroupRef.current = null;
      cameraMarkerGroupRef.current = null;
      selectedCameraMarkerRef.current = null;
    };
  }, [cameraPoses, manifest]);

  useEffect(() => {
    for (const [label, object] of labelObjectsRef.current) {
      object.visible = visibleLabels.has(label);
    }
  }, [visibleLabels]);

  useEffect(() => {
    pickedPointsRef.current = [];
    measurementGroupRef.current?.clear();
  }, [measurementResetKey]);

  useEffect(() => {
    const group = cameraMarkerGroupRef.current;
    if (!group) return;
    if (selectedCameraMarkerRef.current) group.remove(selectedCameraMarkerRef.current);
    const pose = cameraPoses.find((item) => item.frameIndex === selectedFrameIndex);
    if (!pose) {
      selectedCameraMarkerRef.current = null;
      return;
    }
    const marker = makeMarker(scenePosition(pose.x, pose.y, pose.z), "#fcdd78", 0.11);
    group.add(marker);
    selectedCameraMarkerRef.current = marker;
  }, [cameraPoses, selectedFrameIndex]);

  return (
    <div
      className={`viewer-host ${measurementEnabled ? "is-measuring" : ""}`}
      ref={hostRef}
      aria-label="Interactive reconstruction viewport"
      data-testid="point-cloud-viewer"
    >
      <div className="viewport-chrome viewport-chrome--top">
        <span className="live-dot" />
        <span>{manifest.cloud.coordinate_frame.replaceAll("_", " ")}</span>
        <span className="viewport-divider" />
        <span>PLY · sparse evidence</span>
      </div>
      <div className="axis-readout" aria-hidden="true">
        <span className="axis axis--x">E</span>
        <span className="axis axis--y">U</span>
        <span className="axis axis--z">N</span>
      </div>
      <div className="viewport-hint">
        {measurementEnabled ? "Select two observed points" : "Drag to orbit · Scroll to zoom"}
      </div>
      {loadState === "LOADING" && <div className="viewer-state">Loading declared point cloud…</div>}
      {loadState === "ERROR" && (
        <div className="viewer-state viewer-state--error">
          <strong>Cloud unavailable</strong>
          <span>{loadError}</span>
        </div>
      )}
    </div>
  );
}
