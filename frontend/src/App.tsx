import { useEffect, useRef, useState } from "react";

import {
  createSyntheticDemo,
  getProject,
  getRun,
  getViewerManifest,
  loadOfflineFixture,
  loadViewerBundle,
  pollRun,
  startRun,
  uploadProject,
} from "./api";
import { ProgressScreen } from "./components/ProgressScreen";
import { SetupScreen, type UploadInput } from "./components/SetupScreen";
import { Workspace } from "./components/Workspace";
import type { ProjectManifest, RunRecord, ViewerBundle } from "./types";

type Screen = "SETUP" | "PROCESSING" | "WORKSPACE";

export default function App() {
  const [screen, setScreen] = useState<Screen>("SETUP");
  const [project, setProject] = useState<ProjectManifest | null>(null);
  const [run, setRun] = useState<RunRecord | null>(null);
  const [bundle, setBundle] = useState<ViewerBundle | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const pollControllerRef = useRef<AbortController | null>(null);

  const reset = () => {
    pollControllerRef.current?.abort();
    window.history.replaceState(null, "", window.location.pathname);
    setScreen("SETUP");
    setProject(null);
    setRun(null);
    setBundle(null);
    setBusy(false);
    setError("");
  };

  const watchRun = async (initial: RunRecord) => {
    pollControllerRef.current?.abort();
    const controller = new AbortController();
    pollControllerRef.current = controller;
    setRun(initial);
    window.history.replaceState(null, "", `?run=${encodeURIComponent(initial.run_id)}`);
    setScreen("PROCESSING");
    const completed = await pollRun(initial.run_id, setRun, controller.signal);
    if (completed.status === "FAILED") return;
    const manifest = await getViewerManifest(completed.run_id);
    const loaded = await loadViewerBundle(manifest);
    setRun(completed);
    setBundle(loaded);
    setScreen("WORKSPACE");
  };

  const openExistingRun = async (runId: string) => {
    setBusy(true);
    setError("");
    try {
      const existingRun = await getRun(runId);
      const existingProject = await getProject(existingRun.project_id);
      setProject(existingProject);
      await watchRun(existingRun);
    } catch (cause) {
      if (cause instanceof DOMException && cause.name === "AbortError") return;
      setError(cause instanceof Error ? cause.message : "The saved run could not be opened.");
      setScreen("SETUP");
    } finally {
      setBusy(false);
    }
  };

  const handleUpload = async (input: UploadInput) => {
    setBusy(true);
    setError("");
    try {
      const createdProject = await uploadProject(input);
      setProject(createdProject);
      const config: Record<string, unknown> = {
        execution_mode: "COLMAP",
        profile: "preview",
        matcher: "SIFT",
        use_gpu: input.useGpu,
        enable_dense_reconstruction: input.enableDenseReconstruction,
      };
      if (input.preprocessingRun) config.preprocessing_run = input.preprocessingRun;
      if (input.forceIncludeFrameIndices.length) {
        config.force_include_frame_indices = input.forceIncludeFrameIndices;
      }
      if (input.forceExcludeFrameIndices.length) {
        config.force_exclude_frame_indices = input.forceExcludeFrameIndices;
      }
      if (input.knownDistanceM) config.known_distance_m = input.knownDistanceM;
      const createdRun = await startRun(createdProject.project_id, config);
      await watchRun(createdRun);
    } catch (cause) {
      if (cause instanceof DOMException && cause.name === "AbortError") return;
      setError(cause instanceof Error ? cause.message : "The run could not be started.");
      setScreen("SETUP");
    } finally {
      setBusy(false);
    }
  };

  const handleDemo = async () => {
    setBusy(true);
    setError("");
    try {
      const created = await createSyntheticDemo();
      setProject(created.project);
      await watchRun(created.run);
    } catch (cause) {
      if (cause instanceof DOMException && cause.name === "AbortError") return;
      setError(
        `${cause instanceof Error ? cause.message : "Local API unavailable."} You can still open the offline UI fixture.`,
      );
      setScreen("SETUP");
    } finally {
      setBusy(false);
    }
  };

  const handleOfflineFixture = async () => {
    setBusy(true);
    setError("");
    try {
      const loaded = await loadOfflineFixture();
      setBundle(loaded);
      setProject(null);
      setRun(null);
      setScreen("WORKSPACE");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Offline fixture could not be loaded.");
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("fixture") === "1") {
      void handleOfflineFixture();
    } else if (params.get("run")) {
      void openExistingRun(params.get("run")!);
    }
    return () => pollControllerRef.current?.abort();
    // This is intentionally a one-time deep-link check.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <button type="button" className="brand" onClick={reset} aria-label="Trace3D home">
          <span className="brand-mark"><i /><i /><i /></span>
          <span><strong>TRACE</strong><b>3D</b></span>
        </button>
        <div className="header-context">
          <span>RECONSTRUCTION EVIDENCE WORKSPACE</span>
          {run && <code>{run.run_id}</code>}
        </div>
        <div className="system-state">
          <span className="system-dot" />
          <span>LOCAL OPERATOR</span>
          <time>{new Date().toLocaleDateString(undefined, { day: "2-digit", month: "short", year: "numeric" })}</time>
        </div>
      </header>

      {screen === "SETUP" && (
        <SetupScreen
          busy={busy}
          error={error}
          onUpload={(input) => void handleUpload(input)}
          onDemo={() => void handleDemo()}
          onOfflineFixture={() => void handleOfflineFixture()}
        />
      )}
      {screen === "PROCESSING" && run && <ProgressScreen run={run} onReset={reset} />}
      {screen === "WORKSPACE" && bundle && (
        <Workspace bundle={bundle} project={project} run={run} onReset={reset} />
      )}
    </div>
  );
}
