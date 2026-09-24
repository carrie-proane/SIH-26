import { useEffect, useMemo, useState } from "react";

import { getCompletionStatus, getCoverageStatus, requestCompletion } from "../api";
import type { CompletionStatus, CoverageStatus, GapDecision } from "../types";

interface DraftReview {
  decision: GapDecision;
  reviewer: string;
  explanation: string;
  evidencePath: string;
}

export function GapReviewPanel({ runId, onCompleted }: { runId: string; onCompleted: () => void }) {
  const [coverage, setCoverage] = useState<CoverageStatus | null>(null);
  const [completion, setCompletion] = useState<CompletionStatus | null>(null);
  const [drafts, setDrafts] = useState<Record<string, DraftReview>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      getCoverageStatus(runId, controller.signal),
      getCompletionStatus(runId, controller.signal).catch(() => null),
    ]).then(([nextCoverage, nextCompletion]) => {
      setCoverage(nextCoverage);
      setCompletion(nextCompletion);
    }).catch((cause) => {
      if (!(cause instanceof DOMException && cause.name === "AbortError")) {
        setError(cause instanceof Error ? cause.message : "Gap review is unavailable.");
      }
    });
    return () => controller.abort();
  }, [runId]);

  const regions = coverage?.candidate_missing_region_statistics?.regions ?? [];
  const evidence = coverage?.review_evidence_options ?? [];
  const evidenceByPath = useMemo(
    () => new Map(evidence.map((item) => [item.artifact_path, item])),
    [evidence],
  );

  const update = (boundaryId: string, patch: Partial<DraftReview>) => {
    setDrafts((current) => {
      const base: DraftReview = Object.prototype.hasOwnProperty.call(current, boundaryId)
        ? current[boundaryId]
        : {
        decision: "UNKNOWN",
        reviewer: "",
        explanation: "",
        evidencePath: "",
      };
      return {
        ...current,
        [boundaryId]: {
        ...base,
        ...patch,
      },
      };
    });
  };

  const submit = async () => {
    if (!coverage?.source_geometry_sha256) return;
    const reviewed = regions.filter((region) => drafts[region.boundary_id]);
    if (!reviewed.length) { setError("Review at least one candidate region."); return; }
    const reviews = reviewed.map((region) => {
      const draft = drafts[region.boundary_id];
      const chosen = evidenceByPath.get(draft.evidencePath);
      if (!draft.reviewer.trim() || !draft.explanation.trim()) {
        throw new Error("Every decision requires a reviewer and explanation.");
      }
      if (draft.decision === "CONFIRMED_SMALL_GAP" && !chosen) {
        throw new Error("A confirmed gap requires one declared selected-frame image.");
      }
      return {
        boundary_id: region.boundary_id,
        decision: draft.decision,
        reviewer: draft.reviewer.trim(),
        explanation: draft.explanation.trim(),
        source_images: chosen ? [{ path: chosen.artifact_path, sha256: chosen.sha256 }] : [],
      };
    });
    setBusy(true); setError("");
    try {
      const result = await requestCompletion(runId, {
        method: "BOUNDED_PLANAR_GAP",
        source_geometry_sha256: coverage.source_geometry_sha256,
        reviews,
      });
      setCompletion(result);
      onCompleted();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Completion request failed.");
    } finally { setBusy(false); }
  };

  return <section className="inspector-section gap-review-panel">
    <div className="section-title-row"><span>Gap review</span><b>{coverage?.status ?? "LOADING"}</b></div>
    <p className="metric-disclaimer">Only bounded candidates confirmed against declared source images can create inferred patches. Structural openings and unknown regions stay open. Inferred output never becomes measurement eligible.</p>
    {coverage?.metric_depth_status && <small>Metric depth support: {coverage.metric_depth_status}. Topology remains the fallback when valid camera-Z depth is unavailable.</small>}
    {regions.map((region) => {
      const draft = drafts[region.boundary_id] ?? { decision: "UNKNOWN", reviewer: "", explanation: "", evidencePath: "" };
      const geometryBlocked = region.eligible_for_operator_review === false;
      return <details key={region.boundary_id} className="gap-candidate">
        <summary>{region.boundary_id.slice(0, 14)}… · {geometryBlocked ? "geometry rejected" : "reviewable"}</summary>
        <small>{region.boundary_coordinates_enu_m?.length ?? 0} ENU boundary vertices · {region.reasons?.join(", ") || "geometric gates passed"}</small>
        <label className="field"><span>Decision</span><select value={draft.decision} disabled={geometryBlocked || busy} onChange={(event) => update(region.boundary_id, { decision: event.target.value as GapDecision })}><option value="UNKNOWN">Unknown — do not fill</option><option value="STRUCTURAL_OPENING">Structural opening — preserve</option><option value="CONFIRMED_SMALL_GAP">Confirmed small gap — infer patch</option></select></label>
        <label className="field"><span>Reviewer</span><input value={draft.reviewer} disabled={geometryBlocked || busy} onChange={(event) => update(region.boundary_id, { reviewer: event.target.value })} /></label>
        <label className="field"><span>Explanation</span><textarea value={draft.explanation} disabled={geometryBlocked || busy} onChange={(event) => update(region.boundary_id, { explanation: event.target.value })} /></label>
        <label className="field"><span>Declared image evidence</span><select value={draft.evidencePath} disabled={geometryBlocked || busy} onChange={(event) => update(region.boundary_id, { evidencePath: event.target.value })}><option value="">None</option>{evidence.map((item) => <option key={`${item.artifact_path}-${item.sha256}`} value={item.artifact_path}>{item.image_name} · {item.sha256.slice(0, 12)}…</option>)}</select></label>
      </details>;
    })}
    {!regions.length && coverage && <p>No bounded mesh-gap candidates are available for operator review.</p>}
    {regions.length > 0 && <button type="button" className="secondary-action" disabled={busy || !coverage?.source_geometry_sha256} onClick={() => void submit().catch((cause) => setError(cause instanceof Error ? cause.message : "Invalid review"))}>{busy ? "Submitting…" : "Submit reviewed gaps"}</button>}
    {completion && <p className="metric-disclaimer">Completion: {completion.status}. {completion.failure_reason ?? (completion.inferred_regions_present ? "Separate inferred geometry is ready; viewer and export status have refreshed." : "No inferred geometry was produced.")}</p>}
    {error && <p className="inline-error" role="alert">{error}</p>}
  </section>;
}
