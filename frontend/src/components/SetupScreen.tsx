import { useState, type FormEvent } from "react";
import type { ProjectManifest, RunConfiguration, RunRecord } from "../types";

export interface UploadInput {
  name: string; description: string; video: File; telemetry: File; config: RunConfiguration;
}

interface Props {
  busy: boolean; error: string; projects: ProjectManifest[]; selectedProject: ProjectManifest | null; runs: RunRecord[];
  onSelectProject: (id: string) => void; onOpenRun: (id: string) => void; onRefresh: () => void;
  onUpload: (input: UploadInput) => void; onDemo: () => void; onOfflineFixture: () => void;
}

const DEFAULT_CONFIG: RunConfiguration = {
  profile: "preview", matcher: "SIFT", execution_mode: "COLMAP", reconstruction_target: "FULL_SCENE",
  masking_mode: "OFF", enable_segmentation: false, enable_dense_reconstruction: false,
  dense_provider: "auto", use_gpu: false, camera_model: "SIMPLE_RADIAL", camera_model_policy: "AUTO",
  camera_params_reference: "PROCESSED_FRAMES", refine_intrinsics: true,
  force_include_frame_indices: [], force_exclude_frame_indices: [], max_candidate_frames: 240,
  max_selected_frames: 100, worker_threads: 0, sequential_overlap: 10, matching_strategy: "AUTO",
  sparse_timeout_s: 7200, dense_timeout_s: 21600, command_heartbeat_s: 10, coverage_interval_s: 60,
};

function parseIndices(value: string): number[] | null {
  if (!value.trim()) return [];
  const items = value.split(",").map((item) => Number(item.trim()));
  return items.every((item) => Number.isInteger(item) && item >= 0) && new Set(items).size === items.length ? items : null;
}

function UploadGlyph() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M5 14v4.5A1.5 1.5 0 006.5 20h11a1.5 1.5 0 001.5-1.5V14" /></svg>;
}

export function SetupScreen({ busy, error, projects, selectedProject, runs, onSelectProject, onOpenRun, onRefresh, onUpload, onDemo, onOfflineFixture }: Props) {
  const [video, setVideo] = useState<File | null>(null); const [telemetry, setTelemetry] = useState<File | null>(null);
  const [name, setName] = useState("Campus facade run"); const [description, setDescription] = useState("");
  const [config, setConfig] = useState<RunConfiguration>(DEFAULT_CONFIG);
  const [include, setInclude] = useState(""); const [exclude, setExclude] = useState("");
  const [offset, setOffset] = useState(""); const [resolution, setResolution] = useState("");
  const [knownDistance, setKnownDistance] = useState(""); const [localError, setLocalError] = useState("");
  const patch = <K extends keyof RunConfiguration>(key: K, value: RunConfiguration[K]) => setConfig((old) => ({ ...old, [key]: value }));

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (!video || !telemetry) return setLocalError("Choose both the video and its matching telemetry file.");
    const forceInclude = parseIndices(include); const forceExclude = parseIndices(exclude);
    if (!forceInclude || !forceExclude) return setLocalError("Frame overrides must be unique comma-separated non-negative indices.");
    if (forceInclude.some((item) => forceExclude.includes(item))) return setLocalError("A frame cannot be both included and excluded.");
    if (config.max_selected_frames > config.max_candidate_frames) return setLocalError("Selected-frame limit cannot exceed candidate-frame limit.");
    const finalConfig: RunConfiguration = {
      ...config, force_include_frame_indices: forceInclude, force_exclude_frame_indices: forceExclude,
      enable_segmentation: config.masking_mode !== "OFF",
      telemetry_offset_s: offset === "" ? null : Number(offset), telemetry_offset_source: offset === "" ? null : "manual",
      processing_max_image_dimension: resolution === "" ? null : Number(resolution),
      known_distance_m: knownDistance === "" ? null : Number(knownDistance),
    };
    setLocalError(""); onUpload({ name, description, video, telemetry, config: finalConfig });
  };

  return <main className="launch-layout">
    <section className="launch-copy">
      <div className="eyebrow">SIH26158 · server reconstruction evidence</div>
      <h1>See what the camera proved.<span>Question everything else.</span></h1>
      <p>Create a controlled evidence run, inspect sparse results as soon as they exist, and revisit backend-persisted work after refresh.</p>
      <div className="promise-grid"><div><strong>Observed first</strong><span>Generated surfaces remain separate and non-measurable.</span></div><div><strong>Traceable output</strong><span>Every metric and download binds to declared run artifacts.</span></div><div><strong>Server authoritative</strong><span>The API host—not this browser—is the processing machine.</span></div></div>

      <section className="catalog-card" aria-label="Saved projects and runs">
        <div className="section-title-row"><span>Saved server projects</span><button type="button" className="text-action" onClick={onRefresh} disabled={busy}>Refresh</button></div>
        {projects.length ? <label className="field"><span>Project</span><select aria-label="Saved project" value={selectedProject?.project_id ?? ""} onChange={(e) => onSelectProject(e.target.value)}><option value="">Choose a project</option>{projects.map((item) => <option key={item.project_id} value={item.project_id}>{item.name}</option>)}</select></label> : <p className="empty-inline">No server projects yet. Create one below or refresh the catalog.</p>}
        {selectedProject && <div className="saved-runs"><strong>{selectedProject.name}</strong>{runs.length ? runs.map((item) => <button type="button" key={item.run_id} onClick={() => onOpenRun(item.run_id)} disabled={busy}><span>{item.run_id}</span><b>{item.status}</b><small>{item.derived_from_run_id ? `Linked rerun of ${item.derived_from_run_id}` : "Original run"} · {new Date(item.updated_at).toLocaleString()}</small></button>) : <p className="empty-inline">This project has no runs.</p>}</div>}
      </section>
    </section>

    <section className="launch-panel" aria-labelledby="new-run-title">
      <div className="panel-heading"><div><span className="panel-kicker">New evidence run</span><h2 id="new-run-title">Load a controlled capture</h2></div><span className="local-badge"><i /> API</span></div>
      <form onSubmit={submit}>
        <label className="field"><span>Project name</span><input value={name} onChange={(e) => setName(e.target.value)} required /></label>
        <label className="field"><span>Operator note <small>optional</small></span><input value={description} onChange={(e) => setDescription(e.target.value)} placeholder="Scene, weather, capture caveats" /></label>
        <div className="file-pair">
          <label className={`file-drop ${video ? "has-file" : ""}`}><UploadGlyph /><strong>{video?.name ?? "Drone video"}</strong><span>{video ? `${(video.size / 1e6).toFixed(1)} MB` : "MP4 or MOV"}</span><input type="file" accept="video/mp4,video/quicktime,.mp4,.mov" onChange={(e) => setVideo(e.target.files?.[0] ?? null)} /></label>
          <label className={`file-drop ${telemetry ? "has-file" : ""}`}><UploadGlyph /><strong>{telemetry?.name ?? "Flight telemetry"}</strong><span>{telemetry ? `${(telemetry.size / 1000).toFixed(1)} KB` : "DJI SRT or normalized CSV"}</span><input type="file" accept=".srt,.csv,text/csv,application/x-subrip" onChange={(e) => setTelemetry(e.target.files?.[0] ?? null)} /></label>
        </div>
        <div className="compact-fields">
          <label className="field"><span>Processing profile</span><select value={config.profile} onChange={(e) => patch("profile", e.target.value as RunConfiguration["profile"])}>{["smoke","preview","balanced","accurate","diagnostic"].map((v) => <option key={v}>{v}</option>)}</select></label>
          <label className="field"><span>Execution preference</span><select aria-label="Execution preference" disabled><option>COLMAP · SIFT (supported)</option></select></label>
        </div>
        <div className="compact-fields reconstruction-policy-fields">
          <label className="field"><span>Reconstruction target</span><select value={config.reconstruction_target} onChange={(e) => { const target = e.target.value as RunConfiguration["reconstruction_target"]; patch("reconstruction_target", target); if (target === "PRIMARY_SUBJECT") patch("masking_mode", "AUTO"); }}><option value="FULL_SCENE">Full static scene</option><option value="PRIMARY_SUBJECT">Primary subject</option></select></label>
          <label className="field"><span>Scene-aware masking</span><select value={config.masking_mode} onChange={(e) => patch("masking_mode", e.target.value as RunConfiguration["masking_mode"])} disabled={config.reconstruction_target === "PRIMARY_SUBJECT"}><option value="OFF">Off</option><option value="AUTO">Auto with explicit fallback</option><option value="REQUIRED">Required or stop</option></select></label>
        </div>
        <label className="field"><span>Server segmentation weights <small>optional for Auto masking</small></span><input value={config.segmentation_model_path ?? ""} onChange={(e) => patch("segmentation_model_path", e.target.value || null)} placeholder="Server-local weights path; nothing is downloaded automatically" /></label>
        <div className="compact-fields"><label className="check-field"><input type="checkbox" checked={config.enable_dense_reconstruction} onChange={(e) => patch("enable_dense_reconstruction", e.target.checked)} /><span>Request optional dense model</span></label><label className="check-field"><input type="checkbox" checked={config.use_gpu} onChange={(e) => patch("use_gpu", e.target.checked)} /><span>Prefer server GPU</span></label></div>

        <details className="advanced-settings"><summary>Advanced preprocessing options</summary>
          <p className="policy-caveat">SIFT is the only executed matcher. Learned matcher requests are rejected until that backend exists.</p>
          <div className="compact-fields"><label className="field"><span>Telemetry offset <small>seconds, −5 to 5</small></span><input type="number" min="-5" max="5" step="0.001" value={offset} onChange={(e) => setOffset(e.target.value)} /></label><label className="field"><span>Dense provider</span><select value={config.dense_provider} onChange={(e) => patch("dense_provider", e.target.value as RunConfiguration["dense_provider"])}><option value="auto">Automatic capability selection</option><option value="colmap">COLMAP</option><option value="openmvs">OpenMVS</option></select></label></div>
          <div className="compact-fields"><label className="field"><span>Camera model</span><select value={config.camera_model} onChange={(e) => patch("camera_model", e.target.value as RunConfiguration["camera_model"])}><option>SIMPLE_RADIAL</option><option>RADIAL</option><option>OPENCV</option></select></label><label className="field"><span>Camera policy</span><select value={config.camera_model_policy} onChange={(e) => patch("camera_model_policy", e.target.value as RunConfiguration["camera_model_policy"])}><option>AUTO</option><option>FIXED</option></select></label></div>
          <label className="field"><span>Calibration parameters <small>COLMAP format, optional</small></span><input value={config.camera_params ?? ""} onChange={(e) => patch("camera_params", e.target.value || null)} placeholder="e.g. f,cx,cy,k" /></label>
          <label className="check-field"><input type="checkbox" checked={config.refine_intrinsics} onChange={(e) => patch("refine_intrinsics", e.target.checked)} /><span>Refine camera intrinsics</span></label>
          <div className="compact-fields"><label className="field"><span>Candidate frames</span><input type="number" min="3" max="5000" value={config.max_candidate_frames} onChange={(e) => patch("max_candidate_frames", Number(e.target.value))} /></label><label className="field"><span>Selected frames</span><input type="number" min="3" max="1000" value={config.max_selected_frames} onChange={(e) => patch("max_selected_frames", Number(e.target.value))} /></label></div>
          <div className="compact-fields"><label className="field"><span>Max image dimension</span><input type="number" min="640" max="16384" value={resolution} onChange={(e) => setResolution(e.target.value)} placeholder="native" /></label><label className="field"><span>Worker threads <small>0 = server default</small></span><input type="number" min="0" max="256" value={config.worker_threads} onChange={(e) => patch("worker_threads", Number(e.target.value))} /></label></div>
          <div className="compact-fields"><label className="field"><span>Force include frames</span><input value={include} onChange={(e) => setInclude(e.target.value)} placeholder="e.g. 30, 90" /></label><label className="field"><span>Force exclude frames</span><input value={exclude} onChange={(e) => setExclude(e.target.value)} placeholder="e.g. 45" /></label></div>
          <label className="field"><span>External handoff path <small>optional debugging override</small></span><input value={config.preprocessing_run ?? ""} onChange={(e) => patch("preprocessing_run", e.target.value || null)} placeholder="Leave blank for automatic scored preprocessing" /></label>
          <label className="field"><span>Independent known distance <small>metres, optional scale control only</small></span><input type="number" min="0.001" step="0.001" value={knownDistance} onChange={(e) => setKnownDistance(e.target.value)} /></label>
        </details>
        {(localError || error) && <div className="form-error" role="alert">{localError || error}</div>}
        <button className="primary-action" type="submit" disabled={busy}><span>{busy ? "Submitting to API…" : "Create project and run"}</span><b aria-hidden="true">→</b></button>
      </form>
      <div className="demo-divider"><span>or inspect labelled fixtures</span></div><div className="demo-actions"><button type="button" className="demo-button" onClick={onDemo} disabled={busy}><span className="demo-icon">◇</span><span><strong>Run API smoke fixture</strong><small>Synthetic orchestration test, not reconstruction proof</small></span></button><button type="button" className="text-action" onClick={onOfflineFixture} disabled={busy}>Open offline UI fixture</button></div>
    </section>
  </main>;
}
