import { useEffect, useMemo, useState } from "react";

import { createExports, createMeasurement, getExports, getMeasurements, resolveAssetUrl } from "../api";
import type {
  ConfidenceLabel,
  ExportReadiness,
  Keyframe,
  MeasurementRecordPayload,
  MeasurementResult,
  ProjectManifest,
  RunConfiguration,
  RunReadiness,
  RunRecord,
  VisualMode,
  ViewerBundle,
} from "../types";
import {
  visualModeAvailable,
  visualModeMeasurementEligible,
  visualModeReason,
} from "../visualModels";
import { coordinateFramePresentation } from "../coordinateFrame";
import { PointCloudViewer } from "./PointCloudViewer";

interface WorkspaceProps {
  bundle: ViewerBundle;
  project: ProjectManifest | null;
  run: RunRecord | null;
  readiness: RunReadiness | null;
  actionBusy: string;
  globalError: string;
  onReset: () => void;
  onCancel?: () => void;
  onResume?: () => void;
  onRerun: (config: RunConfiguration) => void;
}

const IDLE_MEASUREMENT: MeasurementResult = {
  distanceM: null,
  labels: [],
  status: "IDLE",
  message: "Enable measure, then select two visible points in the cloud.",
};

const UNVERIFIED_VISUAL_ESTIMATE: MeasurementResult = {
  ...IDLE_MEASUREMENT,
  message: "Visual estimate - verification confidence unavailable",
};

function displayPercent(value: number | null | undefined): string {
  return value == null ? "Not evaluated" : `${(value * 100).toFixed(0)}%`;
}

function displayNumber(value: number | null | undefined, suffix = ""): string {
  return value == null ? "Not evaluated" : `${value.toFixed(2)}${suffix}`;
}

function displayBytes(value: number): string {
  return value >= 1_000_000
    ? `${(value / 1_000_000).toFixed(1)} MB`
    : `${(value / 1000).toFixed(1)} KB`;
}

function frameName(frame: Keyframe): string {
  return frame.image_name ?? frame.filename ?? `Frame ${String(frame.frame_index).padStart(4, "0")}`;
}

function SourcePreview({
  frame,
  showDepth,
  showMask,
}: {
  frame: Keyframe | undefined;
  showDepth: boolean;
  showMask: boolean;
}) {
  const base = frame?.image_url;
  const overlay = showDepth ? frame?.depth_overlay_url : showMask ? frame?.mask_url : undefined;
  const mode = showDepth ? "depth" : showMask ? "mask" : "source";
  return (
    <div className={`source-preview ${showDepth ? "is-ai" : ""} ${showMask ? "is-mask" : ""}`}>
      {base && <img src={resolveAssetUrl(base)} alt={`${frameName(frame!)} source evidence`} />}
      {overlay && (
        <img
          className="source-overlay"
          src={resolveAssetUrl(overlay)}
          alt={`${frameName(frame!)} ${mode} overlay`}
        />
      )}
      {!base && !overlay && (
        <div className="source-placeholder" role="img" aria-label="Source frame preview unavailable">
          <span className="scan-line" />
          <b>
            {showDepth
              ? "DEPTH OVERLAY NOT DECLARED"
              : showMask
                ? "MASK OVERLAY NOT DECLARED"
                : "SOURCE PREVIEW NOT DECLARED"}
          </b>
          <small>Frame metadata remains inspectable</small>
        </div>
      )}
      <span className="preview-chip">
        {showDepth
          ? "AI_ASSISTED_NOT_MEASURABLE"
          : showMask
            ? "DYNAMIC MASK OVERLAY"
            : `T +${frame?.timestamp_s.toFixed(2) ?? "—"} s`}
      </span>
    </div>
  );
}

export function Workspace({ bundle, project, run, readiness, actionBusy, globalError, onReset, onCancel, onResume, onRerun }: WorkspaceProps) {
  const { manifest, cameraPoses, keyframes, quality, pointConfidence } = bundle;
  const confidenceAvailable = manifest.confidence.available && pointConfidence !== null;
  const [selectedFrameIndex, setSelectedFrameIndex] = useState<number | null>(
    keyframes[0]?.frame_index ?? null,
  );
  const [visibleLabels, setVisibleLabels] = useState<Set<ConfidenceLabel>>(
    () =>
      new Set(
        confidenceAvailable ? manifest.confidence_legend.map((item) => item.label) : [],
      ),
  );
  const [measurementEnabled, setMeasurementEnabled] = useState(false);
  const [measurement, setMeasurement] = useState<MeasurementResult>(
    confidenceAvailable ? IDLE_MEASUREMENT : UNVERIFIED_VISUAL_ESTIMATE,
  );
  const [measurementResetKey, setMeasurementResetKey] = useState(0);
  const [visualMode, setVisualMode] = useState<VisualMode>("EVIDENCE");
  const [showDepth, setShowDepth] = useState(false);
  const [showMask, setShowMask] = useState(false);
  const [panel, setPanel] = useState<"TRUST" | "SOURCE">("TRUST");
  const [measurements, setMeasurements] = useState<MeasurementRecordPayload[]>([]);
  const [measurementError, setMeasurementError] = useState("");
  const [measurementSaving, setMeasurementSaving] = useState(false);
  const [referenceId, setReferenceId] = useState("");
  const [referenceValue, setReferenceValue] = useState("");
  const [referenceMethod, setReferenceMethod] = useState("");
  const [referenceRole, setReferenceRole] = useState<"NONE" | "SCALE_CONTROL" | "HELD_OUT_EVALUATION">("NONE");
  const [exports, setExports] = useState<ExportReadiness | null>(null);
  const [exportError, setExportError] = useState("");
  const [exportBusy, setExportBusy] = useState(false);
  const [rerunInclude, setRerunInclude] = useState("");
  const [rerunExclude, setRerunExclude] = useState("");

  useEffect(() => {
    if (!run) return;
    const controller = new AbortController();
    void getMeasurements(run.run_id, controller.signal).then((result) => setMeasurements(result.measurements)).catch((cause) => {
      if (!(cause instanceof DOMException && cause.name === "AbortError")) setMeasurementError(cause instanceof Error ? cause.message : "Measurements unavailable.");
    });
    if (readiness?.export_report_ready) {
      void getExports(run.run_id, controller.signal).then(setExports).catch((cause) => {
        if (!(cause instanceof DOMException && cause.name === "AbortError")) setExportError(cause instanceof Error ? cause.message : "Export report unavailable.");
      });
    }
    return () => controller.abort();
  }, [run?.run_id, readiness?.export_report_ready]);

  const selectedFrame = useMemo(
    () => keyframes.find((frame) => frame.frame_index === selectedFrameIndex),
    [keyframes, selectedFrameIndex],
  );

  const toggleLabel = (label: ConfidenceLabel) => {
    setVisibleLabels((previous) => {
      const next = new Set(previous);
      if (next.has(label)) next.delete(label);
      else next.add(label);
      return next;
    });
  };

  const resetMeasurement = () => {
    setMeasurement(confidenceAvailable ? IDLE_MEASUREMENT : UNVERIFIED_VISUAL_ESTIMATE);
    setMeasurementResetKey((value) => value + 1);
  };

  const selectVisualMode = (nextMode: VisualMode) => {
    if (!visualModeAvailable(nextMode, manifest)) return;
    setVisualMode(nextMode);
    setMeasurementEnabled(false);
    resetMeasurement();
  };

  const measurementReference = manifest.measurement_reference;
  const provenance = manifest.source_provenance ?? quality.source_provenance ?? "UNKNOWN";
  const synthetic = provenance === "SYNTHETIC";
  const measurementGeometryEligible = visualModeMeasurementEligible(visualMode, manifest);
  const framePresentation = coordinateFramePresentation(manifest.cloud.coordinate_frame);
  const inputAssets = project?.assets ?? bundle.ingest?.input_assets ?? [];
  const saveMeasurement = async () => {
    if (!run || !measurement.endpoints || !manifest.cloud.relative_path || !manifest.cloud.sha256) return;
    setMeasurementSaving(true); setMeasurementError("");
    try {
      const saved = await createMeasurement(run.run_id, {
        geometry_artifact_path: manifest.cloud.relative_path,
        geometry_artifact_sha256: manifest.cloud.sha256,
        start: measurement.endpoints[0], end: measurement.endpoints[1],
        coordinate_frame: manifest.cloud.coordinate_frame, units: "m", measurement_kind: "DISTANCE_3D",
        reference_id: referenceId.trim() || null,
        reference_value_m: referenceRole === "NONE" || !referenceValue ? null : Number(referenceValue),
        reference_method: referenceRole === "NONE" ? null : referenceMethod.trim() || null,
        reference_role: referenceRole,
      });
      setMeasurements((previous) => [saved, ...previous.filter((item) => item.measurement_id !== saved.measurement_id)]);
    } catch (cause) { setMeasurementError(cause instanceof Error ? cause.message : "Backend rejected the measurement."); }
    finally { setMeasurementSaving(false); }
  };

  const prepareExports = async () => {
    if (!run) return;
    setExportBusy(true); setExportError("");
    try { await createExports(run.run_id); setExports(await getExports(run.run_id)); }
    catch (cause) { setExportError(cause instanceof Error ? cause.message : "Export preparation failed."); }
    finally { setExportBusy(false); }
  };

  const parseRerunIndices = (value: string) => value.trim() ? value.split(",").map((item) => Number(item.trim())) : [];
  const submitRerun = () => {
    if (!run) return;
    const include = parseRerunIndices(rerunInclude); const exclude = parseRerunIndices(rerunExclude);
    if ([...include, ...exclude].some((item) => !Number.isInteger(item) || item < 0) || include.some((item) => exclude.includes(item))) {
      setMeasurementError("Rerun frame overrides must be non-negative integers and cannot overlap."); return;
    }
    onRerun({ ...(run.config as unknown as RunConfiguration), matcher: "SIFT", execution_mode: "COLMAP", force_include_frame_indices: include, force_exclude_frame_indices: exclude });
  };

  return (
    <main className="workspace">
      <aside className="workspace-rail">
        <div className="run-identity">
          <span className="panel-kicker">Active reconstruction</span>
          <h2>{project?.name ?? "Offline viewer fixture"}</h2>
          <code>{manifest.run_id}</code>
          <div className="identity-badges">
            <span className="status-badge status-badge--ready"><i /> {run?.status ?? manifest.status ?? "FIXTURE"}</span>
            <span className={`status-badge status-badge--${provenance.toLowerCase()}`}>
              PROVENANCE: {provenance}
            </span>
          </div>
        </div>

        <div className="rail-section">
          <div className="section-title-row">
            <span>Selected frames</span>
            <b>{keyframes.filter((frame) => frame.selected).length}</b>
          </div>
          <div className="frame-list" aria-label="Frame selection review">
            {keyframes.map((frame) => (
              <button
                type="button"
                className={`${frame.frame_index === selectedFrameIndex ? "is-active" : ""} ${frame.selected ? "" : "is-rejected"}`}
                key={frame.frame_index}
                onClick={() => {
                  setSelectedFrameIndex(frame.frame_index);
                  setPanel("SOURCE");
                }}
              >
                <span className="frame-index">{String(frame.frame_index).padStart(3, "0")}</span>
                <span>
                  <strong>{frameName(frame)}</strong>
                  <small>T +{frame.timestamp_s.toFixed(2)} s</small>
                </span>
                <i className={frame.selected ? "is-selected" : ""} />
              </button>
            ))}
            {!keyframes.length && <div className="empty-inline">No keyframes were declared.</div>}
          </div>
        </div>

        <div className="rail-footer">
          <span>Coordinate frame</span>
          <strong>{manifest.cloud.coordinate_frame}</strong>
          <small>{framePresentation.description}</small>
          <button type="button" onClick={onReset}>← New project</button>
        </div>
      </aside>

      <section className="viewport-section">
        {manifest.preview_mode === "SPARSE_EARLY" && (
          <div className="truth-banner truth-banner--preview"><strong>Early sparse preview</strong><span>Observed sparse evidence is ready; final quality and optional dense results are still pending.</span></div>
        )}
        {readiness && <div className="workspace-readiness" aria-label="Current result readiness"><span>Sparse: {readiness.sparse_preview_ready ? "AVAILABLE" : "NOT READY"}</span><span>Dense: {readiness.dense_status}</span><span>Quality: {readiness.quality_report_ready ? "AVAILABLE" : "PENDING"}</span></div>}
        {synthetic && (
          <div className="truth-banner">
            <strong>UI / orchestration fixture</strong>
            <span>This cloud and its metrics are synthetic. They do not pass the real reconstruction gate.</span>
          </div>
        )}
        <div className="visual-mode-bar" aria-label="Visual reconstruction mode">
          <button
            type="button"
            className={visualMode === "EVIDENCE" ? "is-active" : ""}
            onClick={() => selectVisualMode("EVIDENCE")}
          >
            Observed
          </button>
          <button
            type="button"
            className={visualMode === "INFERRED" ? "is-active" : ""}
            disabled={!visualModeAvailable("INFERRED", manifest)}
            title={visualModeReason("INFERRED", manifest)}
            onClick={() => selectVisualMode("INFERRED")}
          >
            Inferred
          </button>
          <button
            type="button"
            className={visualMode === "BOTH" ? "is-active" : ""}
            disabled={!visualModeAvailable("BOTH", manifest)}
            title={visualModeReason("BOTH", manifest)}
            onClick={() => selectVisualMode("BOTH")}
          >
            Both
          </button>
          <button
            type="button"
            className={visualMode === "TEXTURED" ? "is-active" : ""}
            disabled={!visualModeAvailable("TEXTURED", manifest)}
            title={visualModeReason("TEXTURED", manifest)}
            aria-label={`Textured Model — ${visualModeReason("TEXTURED", manifest)}`}
            onClick={() => selectVisualMode("TEXTURED")}
          >
            Textured Model
          </button>
          <button
            type="button"
            className={visualMode === "PHOTOREAL" ? "is-active" : ""}
            disabled={!visualModeAvailable("PHOTOREAL", manifest)}
            title={visualModeReason("PHOTOREAL", manifest)}
            aria-label={`Photoreal View — ${visualModeReason("PHOTOREAL", manifest)}`}
            onClick={() => selectVisualMode("PHOTOREAL")}
          >
            Photoreal View
          </button>
          <span>{visualModeReason(visualMode, manifest)}</span>
        </div>
        <div className="viewport-toolbar">
          <div className="toolbar-cluster">
            <button type="button" className="tool-button is-active">● Photographic RGB</button>
            <button type="button" className="tool-button is-active">⌁ Flight path</button>
            <button
              type="button"
              className={`tool-button ${showMask ? "is-mask-active" : ""}`}
              disabled={!selectedFrame?.mask_url}
              title={selectedFrame?.mask_url ? "Toggle declared dynamic mask" : "No mask URL declared"}
              onClick={() => {
                setShowMask((value) => !value);
                setShowDepth(false);
                setPanel("SOURCE");
              }}
            >
              ◩ Mask
            </button>
            <button
              type="button"
              className={`tool-button ${showDepth ? "is-ai-active" : ""}`}
              disabled={!manifest.ai_overlay?.available && !selectedFrame?.depth_overlay_url}
              title={manifest.ai_overlay?.reason}
              onClick={() => {
                setShowDepth((value) => !value);
                setShowMask(false);
                setPanel("SOURCE");
              }}
            >
              ◈ AI depth
            </button>
          </div>
          <div className="toolbar-cluster">
            <button
              type="button"
              className={`measure-button ${measurementEnabled ? "is-active" : ""}`}
              disabled={!measurementGeometryEligible}
              title={
                measurementGeometryEligible
                  ? "Measure on evidence geometry"
                  : visualMode === "EVIDENCE"
                    ? "Verified measurement is unavailable for this evidence geometry"
                    : "Measurements are available only on eligible Evidence Cloud geometry"
              }
              onClick={() => {
                if (!measurementGeometryEligible) return;
                setMeasurementEnabled((value) => !value);
                resetMeasurement();
              }}
            >
              <span>↔</span> {measurementEnabled ? "Measuring" : "Measure"}
            </button>
            <button type="button" className="icon-button" onClick={resetMeasurement} title="Clear measurement">↻</button>
          </div>
        </div>
        <PointCloudViewer
          manifest={manifest}
          cameraPoses={cameraPoses}
          pointConfidence={pointConfidence}
          visibleLabels={visibleLabels}
          measurementEnabled={measurementEnabled && measurementGeometryEligible}
          measurementResetKey={measurementResetKey}
          selectedFrameIndex={selectedFrameIndex}
          visualMode={visualMode}
          onMeasurementChange={setMeasurement}
        />
        <div className="measurement-readout" data-status={measurement.status}>
          <span>{measurement.status === "IDLE" ? "DISTANCE TOOL" : measurement.status}</span>
          <strong>{measurement.distanceM === null ? "—" : `${measurement.distanceM.toFixed(3)} m`}</strong>
          <small>
            {measurement.message}
          </small>
          {run && measurement.endpoints && manifest.cloud.measurement_eligible && (
            <div className="measurement-save-controls">
              <label><span>Reference ID <small>optional dataset reference</small></span><input value={referenceId} onChange={(event) => setReferenceId(event.target.value)} placeholder="reference identifier" /></label>
              <label><span>Reference role</span><select value={referenceRole} onChange={(event) => setReferenceRole(event.target.value as typeof referenceRole)}><option value="NONE">Computed only</option><option value="SCALE_CONTROL">Scale control</option><option value="HELD_OUT_EVALUATION">Held-out evaluation</option></select></label>
              {referenceRole !== "NONE" && <><label><span>Independent value <small>metres</small></span><input type="number" min="0.001" step="0.001" required value={referenceValue} onChange={(event) => setReferenceValue(event.target.value)} /></label><label><span>Reference method</span><input value={referenceMethod} onChange={(event) => setReferenceMethod(event.target.value)} placeholder="laser, tape, survey…" /></label></>}
              <button type="button" onClick={() => void saveMeasurement()} disabled={measurementSaving}>{measurementSaving ? "Saving…" : "Save backend measurement"}</button>
            </div>
          )}
          {measurementError && <span className="inline-error" role="alert">{measurementError}</span>}
        </div>
      </section>

      <aside className="inspector">
        <div className="inspector-tabs">
          <button
            type="button"
            className={panel === "TRUST" ? "is-active" : ""}
            onClick={() => setPanel("TRUST")}
          >
            Trust report
          </button>
          <button
            type="button"
            className={panel === "SOURCE" ? "is-active" : ""}
            onClick={() => setPanel("SOURCE")}
          >
            Source frame
          </button>
        </div>

        {panel === "SOURCE" ? (
          <div className="inspector-scroll">
            <SourcePreview frame={selectedFrame} showDepth={showDepth} showMask={showMask} />
            <section className="inspector-section">
              <span className="panel-kicker">Selected evidence</span>
              <h3>{selectedFrame ? frameName(selectedFrame) : "No frame selected"}</h3>
              <dl className="detail-list">
                <div><dt>Frame index</dt><dd>{selectedFrame?.frame_index ?? "—"}</dd></div>
                <div><dt>Timestamp</dt><dd>{selectedFrame ? `${selectedFrame.timestamp_s.toFixed(3)} s` : "—"}</dd></div>
                <div><dt>Blur score</dt><dd>{displayNumber(selectedFrame?.blur_score)}</dd></div>
                <div><dt>Exposure</dt><dd>{displayNumber(selectedFrame?.exposure_score)}</dd></div>
                <div><dt>Dynamic mask</dt><dd>{displayPercent(selectedFrame?.dynamic_mask_fraction)}</dd></div>
                <div><dt>Selection</dt><dd>{selectedFrame?.selected ? "SELECTED" : "REJECTED"}{selectedFrame?.override && selectedFrame.override !== "NONE" ? ` · ${selectedFrame.override}` : ""}</dd></div>
                <div><dt>Quality eligibility</dt><dd>{selectedFrame?.quality_eligible === undefined ? "Not reported" : selectedFrame.quality_eligible ? "ELIGIBLE" : `INELIGIBLE · ${selectedFrame.quality_rejection_reasons || "reason not reported"}`}</dd></div>
                <div><dt>Registration</dt><dd>{selectedFrame?.reconstruction_status ?? "Not evaluated"}</dd></div>
                <div><dt>Exclusion reason</dt><dd>{selectedFrame?.reconstruction_exclusion_reason ?? "—"}</dd></div>
                <div><dt>Source</dt><dd>{selectedFrame?.source ?? "declared artifact"}</dd></div>
              </dl>
            </section>
            {showDepth && (
              <div className="ai-caveat">
                <strong>AI visual assistance only</strong>
                <p>Monocular depth is relative and cannot support the distance tool or verified geometry.</p>
              </div>
            )}
            {showMask && (
              <div className="mask-caveat">
                <strong>Dynamic-content exclusion overlay</strong>
                <p>The mask is review evidence. Missing masks never silently imply a clean frame.</p>
              </div>
            )}
          </div>
        ) : (
          <div className="inspector-scroll">
            <section className="inspector-section trust-summary">
              <div className="section-title-row">
                <span>Run health</span>
                <b className={provenance === "REAL" ? "is-good" : "is-caution"}>
                  {provenance === "REAL" ? "REAL INPUT" : provenance}
                </b>
              </div>
              <div className="metric-grid">
                <div>
                  <span>Registered</span>
                  <strong>{displayPercent(quality.metrics.registered_frame_rate)}</strong>
                  <small>{quality.metrics.registered_frames ?? "—"}/{quality.metrics.eligible_frames ?? "—"} frames</small>
                </div>
                <div>
                  <span>Median reproj.</span>
                  <strong>{displayNumber(quality.metrics.median_reprojection_error_px, " px")}</strong>
                  <small>{quality.metrics.reprojection_gate_1_5_px ? "within gate" : "gate not passed"}</small>
                </div>
                <div>
                  <span>Known distance</span>
                  <strong>{measurementReference.percent_error === null ? "—" : `${measurementReference.percent_error.toFixed(1)}%`}</strong>
                  <small>{measurementReference.passes_10_percent_gate ? "within 10% gate" : "not verified"}</small>
                </div>
                <div>
                  <span>Runtime</span>
                  <strong>{displayNumber(quality.metrics.runtime_s, " s")}</strong>
                  <small>server pipeline</small>
                </div>
              </div>
              {synthetic && (
                <p className="metric-disclaimer">
                  These values only exercise rendering and orchestration. They are not empirical reconstruction results.
                </p>
              )}
            </section>

            {manifest.scene_policy && (
              <section className="inspector-section">
                <div className="section-title-row">
                  <span>Reconstruction policy</span>
                  <b>{manifest.scene_policy.target.replace("_", " ")}</b>
                </div>
                <dl className="detail-list">
                  <div><dt>Target</dt><dd>{manifest.scene_policy.target}</dd></div>
                  <div><dt>Masking</dt><dd>{manifest.scene_policy.masking_mode}</dd></div>
                  <div>
                    <dt>Decision</dt>
                    <dd>
                      {String(
                        (quality.metrics.reconstruction_policy as Record<string, unknown> | undefined)
                          ?.masking_decision ?? "NOT EVALUATED",
                      )}
                    </dd>
                  </div>
                </dl>
                <p className="metric-disclaimer">
                  Masks affect reconstruction only when complete operational mask artifacts are declared.
                </p>
              </section>
            )}

            <section className="inspector-section">
              <div className="section-title-row">
                <span>Input provenance</span>
                <b>{inputAssets.length}</b>
              </div>
              <div className="asset-list">
                {inputAssets.map((asset) => (
                  <div key={`${asset.role}-${asset.sha256}`}>
                    <span>{asset.role}</span>
                    <strong>{asset.original_name}</strong>
                    <small>
                      {displayBytes(asset.size_bytes)} · {asset.origin ?? "UNKNOWN"} · SHA {asset.sha256.slice(0, 12)}…
                    </small>
                  </div>
                ))}
                {!inputAssets.length && <p>No immutable input assets were included in this fixture.</p>}
              </div>
              <div className="probe-note">
                <span>Probe format</span>
                <strong>
                  {String(bundle.ingest?.video_probe?.format?.format_name ?? "not declared")}
                </strong>
              </div>
            </section>

            <section className="inspector-section">
              <div className="section-title-row">
                <span>Confidence layers</span>
                <b>{confidenceAvailable ? `${visibleLabels.size}/${manifest.confidence_legend.length}` : "OFF"}</b>
              </div>
              {!confidenceAvailable && (
                <p className="metric-disclaimer">Confidence unavailable for this run</p>
              )}
              <div className="legend-list">
                {manifest.confidence_legend.map((item) => (
                  <label key={item.label}>
                    <input
                      type="checkbox"
                      checked={visibleLabels.has(item.label)}
                      disabled={!confidenceAvailable}
                      onChange={() => toggleLabel(item.label)}
                    />
                    <i style={{ background: item.color }} />
                    <span>
                      <strong>{item.label.replaceAll("_", " ")}</strong>
                      <small>Measurement: {item.measurement.toLowerCase()}</small>
                    </span>
                  </label>
                ))}
              </div>
            </section>

            <section className="inspector-section known-check">
              <span className="panel-kicker">Independent check</span>
              <h3>{measurementReference.label}</h3>
              <div className="known-values">
                <div><span>Reference</span><strong>{measurementReference.reference_m?.toFixed(3) ?? "—"} m</strong></div>
                <div><span>Reconstructed</span><strong>{measurementReference.measured_m?.toFixed(3) ?? "—"} m</strong></div>
              </div>
              {measurementReference.synthetic_fixture && (
                <p>Fixture values are illustrative and cannot validate scale.</p>
              )}
            </section>

            <section className="inspector-section">
              <div className="section-title-row"><span>Persisted measurements</span><b>{measurements.length}</b></div>
              <p className="metric-disclaimer">The backend distance is authoritative. Confidence describes support, not accuracy; independent evaluation requires a matching dataset reference.</p>
              <div className="asset-list">
                {measurements.map((item) => <div key={item.measurement_id}><span>{item.measurement_kind ?? "DISTANCE_3D"}</span><strong>{item.backend_distance_m.toFixed(3)} {item.units ?? "m"}</strong><small>{item.measurement_eligible ? "Eligible observed geometry" : `Not eligible: ${item.eligibility_reason}`} · {item.geometry_artifact_path} · SHA {item.geometry_artifact_sha256.slice(0, 12)}…{item.reference_id ? ` · reference ${item.reference_id}` : " · computed, not independently evaluated"}</small></div>)}
                {!measurements.length && <p>No saved measurements for this run.</p>}
              </div>
            </section>

            <section className="inspector-section">
              <div className="section-title-row"><span>Geometry exports</span><b>{exports?.available_validated_formats.length ?? 0}</b></div>
              <a className="download-link" href={resolveAssetUrl(manifest.cloud.url)} download>Download observed PLY</a>
              {exports ? <div className="export-list">{[["LAS", "LAS"], ["OBJ", "OBJ"], ["GLB", "GLB_GLTF"], ["GeoTIFF", "GEOTIFF"], ["FBX", "FBX"]].map(([label, key]) => {
                const info = exports.formats[key]; const available = info?.status === "AVAILABLE_VALIDATED";
                return <div key={key}><strong>{label}</strong><span>{info?.status ?? "UNAVAILABLE"}</span>{available && info.exports?.map((item, index) => <div className="export-package" key={`${key}-${index}`}>{key === "OBJ" && item.bundle_url && <a href={resolveAssetUrl(item.bundle_url)} download>Complete OBJ + MTL + textures (.zip)</a>}{item.files.map((file) => <a key={file.relative_path} href={resolveAssetUrl(file.url)} download>{file.relative_path.split("/").at(-1)} · {displayBytes(file.size_bytes)}</a>)}<a href={resolveAssetUrl(item.manifest_url)} download>Provenance manifest</a><small>{String(item.coordinate_contract?.units ?? "units declared in manifest")} · {item.geometry_provenance} · source SHA {item.source_artifact.sha256.slice(0, 12)}…</small></div>)}{!available && <small>{info?.reason ?? "No validated export is declared for this run."}</small>}</div>;
              })}</div> : <p className="metric-disclaimer">No export report is ready. Preparation converts declared terminal geometry only; it does not rerun reconstruction.</p>}
              {run && ["COMPLETED", "FAILED", "CANCELLED"].includes(run.status) && <button type="button" className="secondary-action" onClick={() => void prepareExports()} disabled={exportBusy}>{exportBusy ? "Preparing…" : exports ? "Revalidate exports" : "Prepare exports"}</button>}
              {exportError && <p className="inline-error" role="alert">{exportError}</p>}
            </section>

            <section className="inspector-section quality-sections">
              <div className="section-title-row"><span>Requirement evidence</span><b>{manifest.quality_report_ready ? "FINAL" : "PENDING"}</b></div>
              <dl className="detail-list">
                <div><dt>Registration / reprojection</dt><dd>{quality.metrics.registered_frame_rate == null ? "Not evaluated: final report pending or metric unavailable" : `${displayPercent(quality.metrics.registered_frame_rate)} registered; ${displayNumber(quality.metrics.median_reprojection_error_px, " px")} median`}</dd></div>
                <div><dt>Alignment / synchronization</dt><dd>{quality.metrics.metric_alignment || quality.metrics.telemetry_sync ? "See declared quality report details" : "Not evaluated: no defensible alignment/sync result"}</dd></div>
                <div><dt>Relative-distance evaluation</dt><dd>{measurementReference.percent_error === null ? "Not evaluated: no independent matched reference" : `${measurementReference.percent_error.toFixed(1)}% error on the declared reference`}</dd></div>
                <div><dt>Positional ≤1 m target</dt><dd>Not evaluated: official statistic is unspecified</dd></div>
                <div><dt>Visible-scene coverage</dt><dd>{quality.metrics.coverage ? "Diagnostics declared; registration is not completeness" : "Not evaluated: no coverage artifact"}</dd></div>
                <div><dt>Runtime / resources</dt><dd>{quality.metrics.runtime_s == null ? "Not evaluated: final runtime unavailable" : `${quality.metrics.runtime_s.toFixed(1)} s server pipeline; full-length target requires representative uncached input`}</dd></div>
              </dl>
            </section>

            <section className="inspector-section">
              <div className="section-title-row"><span>Completion boundary</span><b>{manifest.completion?.status ?? "NOT RUN"}</b></div>
              <p className="metric-disclaimer">{manifest.completion?.reason ?? "No completion report or inferred geometry was declared."} Observed geometry remains usable; inferred geometry is separate and never measurement eligible.</p>
            </section>

            {run && <details className="inspector-section"><summary>Run actions and linked rerun</summary>
              <p className="metric-disclaimer">Frame changes create a new linked run; this evidence remains immutable.</p>
              <label className="field"><span>Force include frame indices</span><input value={rerunInclude} onChange={(event) => setRerunInclude(event.target.value)} placeholder="e.g. 12, 18" /></label>
              <label className="field"><span>Force exclude frame indices</span><input value={rerunExclude} onChange={(event) => setRerunExclude(event.target.value)} placeholder="e.g. 20" /></label>
              <div className="run-actions">{run.status === "COMPLETED" ? <button type="button" className="secondary-action" onClick={submitRerun} disabled={Boolean(actionBusy)}>{actionBusy === "rerun" ? "Creating linked run…" : "Create linked rerun"}</button> : <span className="metric-disclaimer">Linked reruns are available after the source run completes.</span>}{onCancel && !["COMPLETED", "FAILED", "CANCELLED"].includes(run.status) && <button type="button" onClick={onCancel} disabled={Boolean(actionBusy)}>Cancel active run</button>}{onResume && ["FAILED", "CANCELLED"].includes(run.status) && <button type="button" onClick={onResume} disabled={Boolean(actionBusy)}>Resume checkpoint</button>}</div>
            </details>}

            {globalError && <div className="form-error" role="alert">{globalError}</div>}

            {!!quality.warnings?.length && (
              <section className="inspector-section">
                <div className="section-title-row"><span>Warnings</span><b>{quality.warnings.length}</b></div>
                <div className="warning-list">
                  {quality.warnings.map((warning) => (
                    <div key={`${warning.code}-${warning.message}`}>
                      <strong>{warning.code}</strong><span>{warning.message}</span>
                    </div>
                  ))}
                </div>
              </section>
            )}

            <section className="inspector-section">
              <div className="section-title-row"><span>Honest limitations</span><b>{quality.limitations.length}</b></div>
              <ul className="limitation-list">
                {quality.limitations.map((limitation) => <li key={limitation}>{limitation}</li>)}
              </ul>
            </section>
          </div>
        )}
      </aside>
    </main>
  );
}
