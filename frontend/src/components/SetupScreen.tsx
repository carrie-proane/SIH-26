import { useState, type FormEvent } from "react";

interface UploadInput {
  name: string;
  description: string;
  video: File;
  telemetry: File;
  preprocessingRun: string;
  forceIncludeFrameIndices: number[];
  forceExcludeFrameIndices: number[];
  knownDistanceM?: number;
  useGpu: boolean;
}

interface SetupScreenProps {
  busy: boolean;
  error: string;
  onUpload: (input: UploadInput) => void;
  onDemo: () => void;
  onOfflineFixture: () => void;
}

function UploadGlyph() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M5 14v4.5A1.5 1.5 0 006.5 20h11a1.5 1.5 0 001.5-1.5V14" />
    </svg>
  );
}

export function SetupScreen({
  busy,
  error,
  onUpload,
  onDemo,
  onOfflineFixture,
}: SetupScreenProps) {
  const [video, setVideo] = useState<File | null>(null);
  const [telemetry, setTelemetry] = useState<File | null>(null);
  const [name, setName] = useState("Campus facade run");
  const [description, setDescription] = useState("");
  const [preprocessingRun, setPreprocessingRun] = useState("");
  const [forceInclude, setForceInclude] = useState("");
  const [forceExclude, setForceExclude] = useState("");
  const [knownDistance, setKnownDistance] = useState("");
  const [useGpu, setUseGpu] = useState(false);
  const [localError, setLocalError] = useState("");

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (!video || !telemetry) {
      setLocalError("Choose both the video and its matching telemetry file.");
      return;
    }
    const parseIndices = (value: string): number[] | null => {
      if (!value.trim()) return [];
      const parsed = value.split(",").map((item) => Number(item.trim()));
      return parsed.every((item) => Number.isInteger(item) && item >= 0) ? parsed : null;
    };
    const included = parseIndices(forceInclude);
    const excluded = parseIndices(forceExclude);
    if (!included || !excluded) {
      setLocalError("Frame overrides must be comma-separated non-negative frame indices.");
      return;
    }
    setLocalError("");
    onUpload({
      name,
      description,
      video,
      telemetry,
      preprocessingRun: preprocessingRun.trim(),
      forceIncludeFrameIndices: included,
      forceExcludeFrameIndices: excluded,
      knownDistanceM: knownDistance ? Number(knownDistance) : undefined,
      useGpu,
    });
  };

  return (
    <main className="launch-layout">
      <section className="launch-copy">
        <div className="eyebrow">SIH26158 · local reconstruction evidence</div>
        <h1>
          See what the camera proved.
          <span>Question everything else.</span>
        </h1>
        <p>
          Turn one controlled drone pass into an inspectable point cloud with source frames,
          flight path, metric checks, and confidence-aware measurement.
        </p>
        <div className="promise-grid">
          <div>
            <strong>Observed first</strong>
            <span>AI depth is visual context, never verified geometry.</span>
          </div>
          <div>
            <strong>Traceable output</strong>
            <span>Every metric links back to a declared run artifact.</span>
          </div>
          <div>
            <strong>Local by design</strong>
            <span>Video and telemetry stay on the operator machine.</span>
          </div>
        </div>
      </section>

      <section className="launch-panel" aria-labelledby="new-run-title">
        <div className="panel-heading">
          <div>
            <span className="panel-kicker">New evidence run</span>
            <h2 id="new-run-title">Load a controlled capture</h2>
          </div>
          <span className="local-badge"><i /> LOCAL</span>
        </div>

        <form onSubmit={submit}>
          <label className="field">
            <span>Project name</span>
            <input value={name} onChange={(event) => setName(event.target.value)} required />
          </label>
          <label className="field">
            <span>Operator note <small>optional</small></span>
            <input
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              placeholder="Scene, weather, capture caveats"
            />
          </label>

          <div className="file-pair">
            <label className={`file-drop ${video ? "has-file" : ""}`}>
              <UploadGlyph />
              <strong>{video?.name ?? "Drone video"}</strong>
              <span>{video ? `${(video.size / 1_000_000).toFixed(1)} MB` : "MP4 or MOV · 30–60 s"}</span>
              <input
                type="file"
                accept="video/mp4,video/quicktime,.mp4,.mov"
                onChange={(event) => setVideo(event.target.files?.[0] ?? null)}
              />
            </label>
            <label className={`file-drop ${telemetry ? "has-file" : ""}`}>
              <UploadGlyph />
              <strong>{telemetry?.name ?? "Flight telemetry"}</strong>
              <span>{telemetry ? `${(telemetry.size / 1000).toFixed(1)} KB` : "DJI SRT or normalized CSV"}</span>
              <input
                type="file"
                accept=".srt,.csv,text/csv,application/x-subrip"
                onChange={(event) => setTelemetry(event.target.files?.[0] ?? null)}
              />
            </label>
          </div>

          <details className="advanced-settings">
            <summary>Advanced preprocessing options</summary>
            <label className="field">
              <span>External handoff path <small>optional debugging override</small></span>
              <input
                value={preprocessingRun}
                onChange={(event) => setPreprocessingRun(event.target.value)}
                placeholder="Leave blank for automatic scored preprocessing"
              />
            </label>
            <div className="compact-fields">
              <label className="field">
                <span>Force include frames <small>comma-separated indices</small></span>
                <input
                  value={forceInclude}
                  onChange={(event) => setForceInclude(event.target.value)}
                  placeholder="e.g. 30, 90"
                />
              </label>
              <label className="field">
                <span>Force exclude frames <small>comma-separated indices</small></span>
                <input
                  value={forceExclude}
                  onChange={(event) => setForceExclude(event.target.value)}
                  placeholder="e.g. 45"
                />
              </label>
            </div>
          </details>
          <div className="compact-fields">
            <label className="field">
              <span>Known distance <small>metres</small></span>
              <input
                type="number"
                min="0.001"
                step="0.001"
                value={knownDistance}
                onChange={(event) => setKnownDistance(event.target.value)}
                placeholder="10.000"
              />
            </label>
            <label className="check-field">
              <input
                type="checkbox"
                checked={useGpu}
                onChange={(event) => setUseGpu(event.target.checked)}
              />
              <span>Use COLMAP GPU</span>
            </label>
          </div>

          {(localError || error) && <div className="form-error" role="alert">{localError || error}</div>}
          <button className="primary-action" type="submit" disabled={busy}>
            <span>{busy ? "Creating run…" : "Start reconstruction"}</span>
            <b aria-hidden="true">→</b>
          </button>
        </form>

        <div className="demo-divider"><span>or inspect the system safely</span></div>
        <div className="demo-actions">
          <button type="button" className="demo-button" onClick={onDemo} disabled={busy}>
            <span className="demo-icon">◇</span>
            <span>
              <strong>Run API smoke fixture</strong>
              <small>Exercises upload → report; synthetic, not reconstruction proof</small>
            </span>
          </button>
          <button type="button" className="text-action" onClick={onOfflineFixture} disabled={busy}>
            Open offline UI fixture
          </button>
        </div>
      </section>
    </main>
  );
}

export type { UploadInput };
