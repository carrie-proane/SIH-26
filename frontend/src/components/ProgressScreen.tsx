import { useEffect, useState } from "react";
import type { RunReadiness, RunRecord, RunStatus } from "../types";

const STAGES: Array<{ status: RunStatus; label: string; detail: string }> = [
  { status: "INGESTING", label: "Ingest", detail: "Checksums, codec and telemetry validation" },
  { status: "PREPROCESSING", label: "Preprocess", detail: "Frame scoring, selection and masks" },
  { status: "RECONSTRUCTING", label: "Reconstruct", detail: "COLMAP poses and sparse evidence" },
  { status: "REPORTING", label: "Report", detail: "Alignment, quality and export contracts" },
  { status: "COMPLETED", label: "Ready", detail: "Declared artifacts available" },
];
const ACTIVE = new Set(["QUEUED", "INGESTING", "PREPROCESSING", "RECONSTRUCTING", "REPORTING"]);
const elapsed = (run: RunRecord, now: number) => {
  const start = Date.parse(run.processing_started_at ?? run.created_at);
  const end = run.processing_completed_at ? Date.parse(run.processing_completed_at) : now;
  if (!Number.isFinite(start) || !Number.isFinite(end)) return "Not available";
  const seconds = Math.max(0, Math.round((end - start) / 1000));
  return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
};

interface Props { run: RunRecord; readiness: RunReadiness | null; error: string; actionBusy: string; onReset: () => void; onCancel: () => void; onResume: () => void; onOpenPreview: () => void; }

export function ProgressScreen({ run, readiness, error, actionBusy, onReset, onCancel, onResume, onOpenPreview }: Props) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => { const id = window.setInterval(() => setNow(Date.now()), 1000); return () => window.clearInterval(id); }, []);
  const stopped = run.status === "FAILED" || run.status === "CANCELLED";
  const activeIndex = Math.max(STAGES.findIndex((item) => item.status === run.stage), 0);
  return <main className="progress-page"><section className="progress-card">
    <div className="eyebrow">Run {run.run_id}{run.derived_from_run_id ? ` · linked from ${run.derived_from_run_id}` : " · original"}</div>
    <div className="progress-title-row"><div><h1>{stopped ? "Run stopped; retained evidence remains available" : "Building reconstruction evidence"}</h1><p>State and timings below come from the API. No browser ETA is inferred.</p></div><div className={`progress-orb ${stopped ? "is-failed" : ""}`}><span className="indeterminate-mark">•••</span><span>{run.stage}</span></div></div>
    <div className="processing-facts"><div><span>Elapsed</span><strong>{elapsed(run, now)}</strong></div><div><span>Requested matcher</span><strong>{run.requested_matcher ?? String(run.config.matcher ?? "not reported")}</strong></div><div><span>Executed matcher</span><strong>{run.executed_matcher ?? "Not executed yet"}</strong></div><div><span>Server GPU</span><strong>{run.effective_sparse_gpu ? "USED" : "NOT REPORTED / CPU"}</strong></div></div>
    <div className="progress-track" aria-label="Processing stages">{STAGES.map((stage, index) => { const failed = stopped && index === activeIndex; const state = failed ? "failed" : index < activeIndex ? "complete" : index === activeIndex ? "active" : "pending"; return <div className={`progress-stage is-${state}`} key={stage.status}><div className="stage-marker">{state === "complete" ? "✓" : index + 1}</div><div><strong>{stage.label}</strong><span>{stage.detail}</span>{run.stage_timings_s?.[stage.label.toUpperCase()] !== undefined && <small>{run.stage_timings_s[stage.label.toUpperCase()].toFixed(1)} s</small>}</div></div>; })}</div>
    <section className="readiness-panel" aria-label="Artifact readiness"><div><span>Sparse preview</span><strong>{readiness?.sparse_preview_ready ? "AVAILABLE" : "NOT READY"}</strong><small>{readiness?.sparse_preview_ready ? "Observed sparse geometry can be inspected now." : readiness?.sparse_preview_missing.join(", ") || "Waiting for declared artifacts."}</small></div><div><span>Dense visual model</span><strong>{readiness?.dense_status ?? "NOT READY"}</strong><small>Optional; sparse evidence is retained if dense processing fails.</small></div><div><span>Quality report</span><strong>{readiness?.quality_report_ready ? "AVAILABLE" : "PENDING"}</strong><small>Early preview does not imply final quality evaluation.</small></div></section>
    {run.failure_reason && <div className="failure-card" role="alert"><span>Run failure</span><strong>{run.failure_reason}</strong><small>{run.artifacts.length} completed artifact(s) retained.</small></div>}
    {error && <div className="form-error" role="alert">{error}</div>}
    <details className="event-log" open><summary className="event-log__header"><strong>Run events</strong><span>live from API</span></summary>{run.events.length ? run.events.slice().reverse().map((event, i) => <div className="event-row" key={`${event.timestamp}-${i}`}><time>{new Date(event.timestamp).toLocaleTimeString()}</time><b>{event.stage}</b><span>{event.message}</span></div>) : <div className="event-empty">Queued. Waiting for the worker…</div>}</details>
    <div className="run-actions">{readiness?.sparse_preview_ready && <button type="button" className="primary-action" onClick={onOpenPreview} disabled={Boolean(actionBusy)}>{actionBusy === "preview" ? "Opening…" : "Open sparse preview"}</button>}{ACTIVE.has(run.status) && <button type="button" className="secondary-action" disabled={Boolean(actionBusy) || Boolean(run.cancel_requested_at)} onClick={onCancel}>{run.cancel_requested_at ? "Cancellation requested; awaiting backend…" : actionBusy === "cancel" ? "Requesting…" : "Cancel run"}</button>}{stopped && <button type="button" className="secondary-action" disabled={Boolean(actionBusy)} onClick={onResume}>{actionBusy === "resume" ? "Resuming…" : "Resume from valid checkpoint"}</button>}<button type="button" className="text-action" onClick={onReset}>Back to projects</button></div>
  </section></main>;
}
