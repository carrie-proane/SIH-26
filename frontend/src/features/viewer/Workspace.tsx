import { useMemo, useState } from "react";

import { resolveAssetUrl } from "../../services/api";
import type {
  ConfidenceLabel,
  Keyframe,
  MeasurementResult,
  ProjectManifest,
  RunRecord,
  VisualMode,
  ViewerBundle,
} from "../../domain/contracts";
import {
  visualModeAvailable,
  visualModeMeasurementEligible,
  visualModeReason,
} from "./visualModels";
import { PointCloudViewer } from "./PointCloudViewer";

interface WorkspaceProps {
  bundle: ViewerBundle;
  project: ProjectManifest | null;
  run: RunRecord | null;
  onReset: () => void;
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

function displayPercent(value: number | undefined): string {
  return value === undefined ? "—" : `${(value * 100).toFixed(0)}%`;
}

function displayNumber(value: number | undefined, suffix = ""): string {
  return value === undefined ? "—" : `${value.toFixed(2)}${suffix}`;
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

export function Workspace({ bundle, project, run, onReset }: WorkspaceProps) {
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
  const inputAssets = project?.assets ?? bundle.ingest?.input_assets ?? [];
  const liveReferenceError =
    measurement.distanceM !== null && measurementReference.reference_m
      ? (Math.abs(measurement.distanceM - measurementReference.reference_m) /
          measurementReference.reference_m) *
        100
      : null;

  return (
    <main className="workspace">
      <aside className="workspace-rail">
        <div className="run-identity">
          <span className="panel-kicker">Active reconstruction</span>
          <h2>{project?.name ?? "Offline viewer fixture"}</h2>
          <code>{manifest.run_id}</code>
          <div className="identity-badges">
            <span className="status-badge status-badge--ready"><i /> COMPLETED</span>
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
          <button type="button" onClick={onReset}>← New project</button>
        </div>
      </aside>

      <section className="viewport-section">
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
            Evidence Cloud
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
            className={visualMode === "PREDICTED" ? "is-active" : ""}
            disabled={!visualModeAvailable("PREDICTED", manifest)}
            title={visualModeReason("PREDICTED", manifest)}
            aria-label={`AI Predicted Surface - ${visualModeReason("PREDICTED", manifest)}`}
            onClick={() => selectVisualMode("PREDICTED")}
          >
            AI Predicted Surface
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
                  : "Measurements are available only on the Evidence Cloud"
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
            {liveReferenceError !== null
              ? ` Interactive error: ${liveReferenceError.toFixed(1)}% vs ${measurementReference.reference_m?.toFixed(3)} m reference.`
              : ""}
          </small>
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
                  <small>local pipeline</small>
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
