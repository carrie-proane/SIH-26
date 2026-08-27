import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";

import {
  declaredVisualArtifactUrls,
  fetchDeclaredVisualArtifact,
  formatByteSize,
  isDeclaredVisualArtifact,
  modelFormat,
  visualModelLabel,
} from "../modelLoading";
import {
  cameraDistanceForSphere,
  pointSizeForRadius,
  robustSceneBounds,
} from "../viewerBounds";
import type {
  CameraPose,
  ConfidenceLabel,
  MeasurementResult,
  PointConfidenceArtifact,
  VisualMode,
  ViewerManifest,
} from "../types";

interface PointCloudViewerProps {
  manifest: ViewerManifest;
  cameraPoses: CameraPose[];
  pointConfidence: PointConfidenceArtifact | null;
  visibleLabels: Set<ConfidenceLabel>;
  measurementEnabled: boolean;
  measurementResetKey: number;
  selectedFrameIndex: number | null;
  visualMode: VisualMode;
  onMeasurementChange: (result: MeasurementResult) => void;
}

interface PickedPoint {
  position: THREE.Vector3;
  label?: ConfidenceLabel;
}

const DISABLED_LABELS = new Set<ConfidenceLabel>([
  "AI_ASSISTED_NOT_MEASURABLE",
  "UNSEEN",
]);

function scenePosition(x: number, y: number, z: number): THREE.Vector3 {
  return new THREE.Vector3(x, z, -y);
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
  pointConfidence,
  visibleLabels,
  measurementEnabled,
  measurementResetKey,
  selectedFrameIndex,
  visualMode,
  onMeasurementChange,
}: PointCloudViewerProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const labelObjectsRef = useRef<Map<ConfidenceLabel, THREE.Points>>(new Map());
  const pointObjectsRef = useRef<THREE.Points[]>([]);
  const confidenceReadyRef = useRef(false);
  const measurementGroupRef = useRef<THREE.Group | null>(null);
  const cameraMarkerGroupRef = useRef<THREE.Group | null>(null);
  const selectedCameraMarkerRef = useRef<THREE.Mesh | null>(null);
  const pickedPointsRef = useRef<PickedPoint[]>([]);
  const measurementEnabledRef = useRef(measurementEnabled);
  const onMeasurementChangeRef = useRef(onMeasurementChange);
  const visibleLabelsRef = useRef(visibleLabels);
  const [loadState, setLoadState] = useState<"LOADING" | "READY" | "ERROR">("LOADING");
  const [loadError, setLoadError] = useState("");
  const [loadProgress, setLoadProgress] = useState({ loaded: 0, total: null as number | null });
  const [loadedBytes, setLoadedBytes] = useState<number | null>(null);
  const [loadNotice, setLoadNotice] = useState("");
  const [loadedModelMode, setLoadedModelMode] = useState<VisualMode>(visualMode);
  const [fallbackUsed, setFallbackUsed] = useState(false);

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

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 5000);
    camera.position.set(7, 6, 8);
    let fittedSphere: THREE.Sphere | null = null;

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

    const fitCamera = (useDefaultDirection = false) => {
      if (!fittedSphere) return;
      const radius = Math.max(fittedSphere.radius, 1);
      const direction = useDefaultDirection
        ? new THREE.Vector3(0.9, 0.65, 1).normalize()
        : camera.position.clone().sub(controls.target).normalize();
      const distance = cameraDistanceForSphere(radius, camera.fov, camera.aspect);
      controls.target.copy(fittedSphere.center);
      camera.position.copy(fittedSphere.center).add(direction.multiplyScalar(distance));
      camera.near = Math.max(radius / 1000, 0.01);
      camera.far = Math.max(radius * 50, 500);
      controls.minDistance = Math.max(radius / 100, 0.05);
      controls.maxDistance = Math.max(radius * 20, 500);
      camera.updateProjectionMatrix();
      controls.update();
    };

    const resize = () => {
      const width = Math.max(host.clientWidth, 1);
      const height = Math.max(host.clientHeight, 1);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      fitCamera();
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
      const objects = pointObjectsRef.current.filter((item) => item.visible);
      const intersection = raycaster.intersectObjects(objects, false)[0];
      if (!intersection?.point) return;
      const label = confidenceReadyRef.current
        ? (intersection.object.userData.confidence as ConfidenceLabel)
        : undefined;
      if (label && DISABLED_LABELS.has(label)) {
        onMeasurementChangeRef.current({
          distanceM: null,
          labels: label ? [label] : [],
          status: "BLOCKED",
          message: `${label} geometry is not eligible for confidence-qualified measurement.`,
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
          labels: label ? [label] : [],
          status: "SELECTING",
          message: confidenceReadyRef.current
            ? "First confidence-qualified point selected. Choose a second point."
            : "Visual estimate - verification confidence unavailable. Choose a second point.",
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
      const labels = [first.label, second.label].filter(
        (value): value is ConfidenceLabel => value !== undefined,
      );
      if (!confidenceReadyRef.current) {
        onMeasurementChangeRef.current({
          distanceM: distance,
          labels: [],
          status: "CAUTION",
          message: "Visual estimate - verification confidence unavailable",
        });
        return;
      }
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
            ? "Both points have explicit high-confidence observed support."
            : status === "CONFIRM"
              ? "Low-confidence geometry requires explicit operator confirmation."
              : "Measurement includes medium-confidence geometry; use with caution.",
      });
    };
    renderer.domElement.addEventListener("pointerdown", onPointerDown);

    const abortController = new AbortController();
    const declaredUrls = declaredVisualArtifactUrls(manifest);
    setLoadState("LOADING");
    setLoadError("");
    setLoadNotice("");
    setLoadProgress({ loaded: 0, total: null });
    setLoadedBytes(null);
    setLoadedModelMode(visualMode);
    setFallbackUsed(false);

    const modelForMode = (mode: VisualMode) => {
      if (mode === "EVIDENCE") {
        return manifest.visual_models?.evidence_cloud ?? {
          available: true,
          url: manifest.cloud.url,
          format: "PLY" as const,
          coordinate_frame: manifest.cloud.coordinate_frame,
          measurement_eligible: true,
        };
      }
      return mode === "TEXTURED"
        ? manifest.visual_models?.textured_mesh
        : manifest.visual_models?.gaussian_splat;
    };

    const fitLoadedObject = (object: THREE.Object3D, positions?: THREE.BufferAttribute) => {
      const bounds = positions
        ? robustSceneBounds(positions)
        : new THREE.Box3().setFromObject(object);
      for (const point of cameraPathPoints) bounds.expandByPoint(point);
      if (bounds.isEmpty()) return;
      const center = bounds.getCenter(new THREE.Vector3());
      const radius = Math.max(bounds.getBoundingSphere(new THREE.Sphere()).radius, 1);
      for (const points of pointObjectsRef.current) {
        if (points.material instanceof THREE.PointsMaterial) {
          points.material.size = pointSizeForRadius(radius);
          points.material.needsUpdate = true;
        }
      }
      fittedSphere = new THREE.Sphere(center, radius);
      fitCamera(true);
    };

    const loadTexture = async (textureUrl: string): Promise<THREE.Texture> => {
      if (!isDeclaredVisualArtifact(manifest, textureUrl)) {
        throw new Error("Refusing to load an undeclared texture artifact.");
      }
      const { buffer } = await fetchDeclaredVisualArtifact(
        manifest,
        textureUrl,
        (loaded, total) => setLoadProgress({ loaded, total }),
        abortController.signal,
      );
      const objectUrl = URL.createObjectURL(new Blob([buffer]));
      try {
        const texture = await new THREE.TextureLoader().loadAsync(objectUrl);
        texture.colorSpace = THREE.SRGBColorSpace;
        return texture;
      } finally {
        URL.revokeObjectURL(objectUrl);
      }
    };

    const buildPlyObject = async (
      geometry: THREE.BufferGeometry,
      targetMode: VisualMode,
      visualModel: ReturnType<typeof modelForMode>,
    ): Promise<{ object: THREE.Group; positions: THREE.BufferAttribute }> => {
      const position = geometry.getAttribute("position");
      if (!position) throw new Error("Declared PLY has no vertex positions.");
      const color = geometry.getAttribute("color");
      const renderedPositions: number[] = [];
      for (let index = 0; index < position.count; index += 1) {
        renderedPositions.push(position.getX(index), position.getZ(index), -position.getY(index));
      }
      const renderedPosition = new THREE.Float32BufferAttribute(renderedPositions, 3);
      const pointGroup = new THREE.Group();
      const explicitConfidence =
        targetMode === "EVIDENCE" &&
        pointConfidence !== null &&
        pointConfidence.points.length === position.count;
      confidenceReadyRef.current = explicitConfidence;
      if (targetMode === "TEXTURED") {
        pointGroup.name = "textured-visual-model-not-measurement-evidence";
        const meshGeometry = new THREE.BufferGeometry();
        meshGeometry.setAttribute("position", renderedPosition);
        if (geometry.index) meshGeometry.setIndex(geometry.index.clone());
        const uv = geometry.getAttribute("uv");
        if (uv) meshGeometry.setAttribute("uv", uv.clone());
        if (color) meshGeometry.setAttribute("color", color.clone());
        meshGeometry.computeVertexNormals();
        let texture: THREE.Texture | null = null;
        const textureUrl =
          targetMode === "TEXTURED"
            ? manifest.visual_models?.textured_mesh?.texture_urls?.[0]
            : undefined;
        if (textureUrl) {
          try {
            texture = await loadTexture(textureUrl);
          } catch (cause) {
            if (cause instanceof DOMException && cause.name === "AbortError") throw cause;
            setLoadNotice(
              `Texture atlas unavailable (${cause instanceof Error ? cause.message : "unknown error"}). ` +
                "Showing the declared mesh with its available vertex colours.",
            );
          }
        }
        const mesh = new THREE.Mesh(
          meshGeometry,
          new THREE.MeshStandardMaterial({
            map: texture,
            color: texture || color ? "#ffffff" : "#aabbb5",
            vertexColors: Boolean(color),
            side: THREE.DoubleSide,
            roughness: 0.9,
          }),
        );
        pointGroup.add(mesh);
      } else if (explicitConfidence) {
        pointGroup.name = "explicit-confidence-point-cloud";
        const grouped = new Map<ConfidenceLabel, number[]>();
        for (const item of manifest.confidence_legend) grouped.set(item.label, []);
        for (const point of pointConfidence.points) {
          grouped.get(point.confidence_class)?.push(
            renderedPosition.getX(point.point_id),
            renderedPosition.getY(point.point_id),
            renderedPosition.getZ(point.point_id),
          );
        }
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
          pointObjectsRef.current.push(points);
          pointGroup.add(points);
        }
      } else {
        pointGroup.name = "photographic-rgb-point-cloud";
        const photographicGeometry = new THREE.BufferGeometry();
        photographicGeometry.setAttribute("position", renderedPosition);
        if (color) photographicGeometry.setAttribute("color", color.clone());
        const points = new THREE.Points(
          photographicGeometry,
          new THREE.PointsMaterial({
            color: color ? "#ffffff" : "#b8c6c2",
            vertexColors: Boolean(color),
            size: 0.13,
            sizeAttenuation: true,
            transparent: true,
            opacity: 0.95,
          }),
        );
        pointObjectsRef.current.push(points);
        pointGroup.add(points);
      }
      return { object: pointGroup, positions: renderedPosition };
    };

    const parseGlb = (buffer: ArrayBuffer): Promise<THREE.Group> =>
      new Promise((resolve, reject) => {
        const manager = new THREE.LoadingManager();
        manager.setURLModifier((url) => {
          if (url.startsWith("data:") || url.startsWith("blob:")) return url;
          throw new Error("GLB references an undeclared external asset.");
        });
        new GLTFLoader(manager).parse(buffer, "", (gltf) => resolve(gltf.scene), reject);
      });

    const loadRequested = async (targetMode: VisualMode, allowFallback: boolean): Promise<void> => {
      const visualModel = modelForMode(targetMode);
      const modelUrl = visualModel?.url ?? (targetMode === "EVIDENCE" ? manifest.cloud.url : null);
      const label = visualModelLabel(targetMode);
      try {
        if (!modelUrl || !declaredUrls.has(modelUrl) || !isDeclaredVisualArtifact(manifest, modelUrl)) {
          throw new Error(`${label} is not declared by this run's viewer manifest.`);
        }
        if (visualModel?.available === false) {
          throw new Error(visualModel.statement ?? `${label} is unavailable for this run.`);
        }
        pointObjectsRef.current = [];
        labelObjectsRef.current.clear();
        confidenceReadyRef.current = false;
        setLoadedModelMode(targetMode);
        const format = modelFormat(visualModel, modelUrl);
        const { buffer, totalBytes } = await fetchDeclaredVisualArtifact(
          manifest,
          modelUrl,
          (loaded, total) => setLoadProgress({ loaded, total }),
          abortController.signal,
        );
        if (disposed) return;
        setLoadedBytes(totalBytes ?? buffer.byteLength);
        let loaded: { object: THREE.Object3D; positions?: THREE.BufferAttribute };
        if (format === "PLY") {
          const geometry = new PLYLoader().parse(buffer);
          const ply = await buildPlyObject(geometry, targetMode, visualModel);
          loaded = ply;
          geometry.dispose();
        } else if (format === "GLB") {
          const glb = await parseGlb(buffer);
          glb.rotation.x = -Math.PI / 2;
          loaded = { object: glb };
          confidenceReadyRef.current = false;
        } else {
          throw new Error("Gaussian Splat artifacts are not supported by this browser build.");
        }
        if (disposed) return;
        scene.add(loaded.object);
        fitLoadedObject(loaded.object, loaded.positions);
        setLoadState("READY");
      } catch (cause) {
        if (disposed || (cause instanceof DOMException && cause.name === "AbortError")) return;
        const reason = cause instanceof Error ? cause.message : "The declared visual artifact could not be parsed.";
        if (allowFallback && targetMode !== "EVIDENCE") {
          setFallbackUsed(true);
          setLoadNotice(`${label} unavailable (${reason}). Showing the declared Evidence Cloud instead.`);
          await loadRequested("EVIDENCE", false);
          return;
        }
        setLoadError(reason);
        setLoadState("ERROR");
      }
    };

    void loadRequested(visualMode, true);

    const animate = () => {
      controls.update();
      renderer.render(scene, camera);
      animationFrame = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      disposed = true;
      abortController.abort();
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
      pointObjectsRef.current = [];
      confidenceReadyRef.current = false;
      measurementGroupRef.current = null;
      cameraMarkerGroupRef.current = null;
      selectedCameraMarkerRef.current = null;
    };
  }, [cameraPoses, manifest, pointConfidence, visualMode]);

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
        <span>{(loadedModelMode === "TEXTURED"
          ? manifest.visual_models?.textured_mesh?.coordinate_frame
          : manifest.cloud.coordinate_frame)?.replaceAll("_", " ")}</span>
        <span className="viewport-divider" />
        <span>{visualModelLabel(loadedModelMode)}</span>
        <span className="viewport-divider" />
        <span>{loadedModelMode === "EVIDENCE" ? "MEASUREMENT ELIGIBLE" : "VISUAL ONLY"}</span>
        <span className="viewport-divider" />
        <span>{manifest.source_provenance} INPUT</span>
      </div>
      <div className="axis-readout" aria-hidden="true">
        <span className="axis axis--x">E</span>
        <span className="axis axis--y">U</span>
        <span className="axis axis--z">N</span>
      </div>
      <div className="viewport-hint">
        {measurementEnabled
          ? pointConfidence
            ? "Select two confidence-qualified points"
            : "Select two visible points"
          : "Drag to orbit · Scroll to zoom"}
      </div>
      {loadNotice && <div className="viewer-notice" role="status">{loadNotice}</div>}
      {loadState === "LOADING" && (
        <div className="viewer-state">
          <strong>Loading {visualModelLabel(visualMode)}…</strong>
          <span>
            {loadProgress.total
              ? `${Math.min(100, Math.round((loadProgress.loaded / loadProgress.total) * 100))}% · `
              : "Streaming · "}
            {formatByteSize(loadProgress.loaded)}
            {loadProgress.total ? ` / ${formatByteSize(loadProgress.total)}` : ""}
          </span>
          <progress
            max={loadProgress.total ?? undefined}
            value={loadProgress.total ? Math.min(loadProgress.loaded, loadProgress.total) : undefined}
            aria-label="Visual model loading progress"
          />
        </div>
      )}
      {loadState === "ERROR" && (
        <div className="viewer-state viewer-state--error">
          <strong>{visualModelLabel(visualMode)} unavailable</strong>
          <span>{loadError}</span>
          {loadedBytes !== null && <small>Artifact size: {formatByteSize(loadedBytes)}</small>}
        </div>
      )}
      {loadState === "READY" && loadedBytes !== null && (
        <div className="viewer-file-meta" aria-label="Loaded visual artifact details">
          <span>{fallbackUsed ? "FALLBACK · " : ""}{visualModelLabel(loadedModelMode)}</span>
          <small>{formatByteSize(loadedBytes)} · {loadedModelMode === "EVIDENCE" ? "measurable" : "visual only"}</small>
        </div>
      )}
    </div>
  );
}
