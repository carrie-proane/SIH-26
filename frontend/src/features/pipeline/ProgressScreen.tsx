import type { RunRecord, RunStatus } from "../../domain/contracts";

const STAGES: Array<{ status: RunStatus; label: string; detail: string }> = [
  { status: "INGESTING", label: "Ingest", detail: "Checksums, codec and telemetry validation" },
  { status: "PREPROCESSING", label: "Preprocess", detail: "Frames, scores, selections and masks" },
  { status: "RECONSTRUCTING", label: "Reconstruct", detail: "COLMAP poses and sparse evidence" },
  { status: "REPORTING", label: "Report", detail: "Alignment, metric checks and limitations" },
  { status: "COMPLETED", label: "Ready", detail: "Declared artifacts available to viewer" },
];

interface ProgressScreenProps {
  run: RunRecord;
  onReset: () => void;
}

export function ProgressScreen({ run, onReset }: ProgressScreenProps) {
  const activeIndex = run.status === "FAILED"
    ? Math.max(STAGES.findIndex((item) => item.status === run.stage), 0)
    : STAGES.findIndex((item) => item.status === run.stage);

  return (
    <main className="progress-page">
      <section className="progress-card">
        <div className="eyebrow">Run {run.run_id}</div>
        <div className="progress-title-row">
          <div>
            <h1>{run.status === "FAILED" ? "Run stopped honestly" : "Building reconstruction evidence"}</h1>
            <p>
              {run.status === "FAILED"
                ? "Completed artifacts and the exact failure reason are retained below."
                : "This page reports backend state directly; progress is never faked by the browser."}
            </p>
          </div>
          <div className={`progress-orb ${run.status === "FAILED" ? "is-failed" : ""}`}>
            <strong>{run.progress}%</strong>
            <span>{run.stage}</span>
          </div>
        </div>

        <div className="progress-track" aria-label="Processing stages">
          {STAGES.map((stage, index) => {
            const failedHere = run.status === "FAILED" && index === activeIndex;
            const state = failedHere ? "failed" : index < activeIndex ? "complete" : index === activeIndex ? "active" : "pending";
            return (
              <div className={`progress-stage is-${state}`} key={stage.status}>
                <div className="stage-marker">{state === "complete" ? "✓" : index + 1}</div>
                <div>
                  <strong>{stage.label}</strong>
                  <span>{stage.detail}</span>
                </div>
              </div>
            );
          })}
        </div>

        {run.failure_reason && (
          <div className="failure-card" role="alert">
            <span>Run failure</span>
            <strong>{run.failure_reason}</strong>
            <small>{run.artifacts.length} completed artifact(s) retained for diagnosis.</small>
          </div>
        )}

        <div className="event-log">
          <div className="event-log__header">
            <strong>Run events</strong>
            <span>live from local API</span>
          </div>
          {run.events.length ? run.events.slice().reverse().map((event, index) => (
            <div className="event-row" key={`${event.timestamp}-${index}`}>
              <time>{new Date(event.timestamp).toLocaleTimeString()}</time>
              <b>{event.stage}</b>
              <span>{event.message}</span>
            </div>
          )) : <div className="event-empty">Queued. Waiting for the pipeline worker…</div>}
        </div>

        {run.status === "FAILED" && (
          <button type="button" className="secondary-action" onClick={onReset}>Start another run</button>
        )}
      </section>
    </main>
  );
}
