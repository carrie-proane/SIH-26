import { useCallback, useEffect, useRef, useState } from "react";

import {
  cancelRun, createSyntheticDemo, getProject, getProjectRuns, getProjects, getRun,
  getRunReadiness, getViewerManifest, loadOfflineFixture, loadViewerBundle, rerunRun,
  resumeRun, startRun, uploadProject,
} from "./api";
import { ProgressScreen } from "./components/ProgressScreen";
import { SetupScreen, type UploadInput } from "./components/SetupScreen";
import { Workspace } from "./components/Workspace";
import type { ProjectManifest, RunConfiguration, RunReadiness, RunRecord, ViewerBundle } from "./types";

type Screen = "SETUP" | "PROCESSING" | "WORKSPACE";
const TERMINAL = new Set(["COMPLETED", "FAILED", "CANCELLED"]);
const isAbort = (cause: unknown) => cause instanceof DOMException && cause.name === "AbortError";

export default function App() {
  const [screen, setScreen] = useState<Screen>("SETUP");
  const [projects, setProjects] = useState<ProjectManifest[]>([]);
  const [projectRuns, setProjectRuns] = useState<RunRecord[]>([]);
  const [project, setProject] = useState<ProjectManifest | null>(null);
  const [run, setRun] = useState<RunRecord | null>(null);
  const [readiness, setReadiness] = useState<RunReadiness | null>(null);
  const [bundle, setBundle] = useState<ViewerBundle | null>(null);
  const [busy, setBusy] = useState(false);
  const [actionBusy, setActionBusy] = useState("");
  const [error, setError] = useState("");
  const requestRef = useRef<AbortController | null>(null);
  const selectionRef = useRef(0);

  const setUrl = (projectId?: string, runId?: string) => {
    const params = new URLSearchParams();
    if (projectId) params.set("project", projectId);
    if (runId) params.set("run", runId);
    window.history.replaceState(null, "", params.size ? `?${params}` : window.location.pathname);
  };

  const refreshCatalog = useCallback(async (projectId?: string, signal?: AbortSignal) => {
    const result = await getProjects(signal);
    setProjects(result.projects);
    if (projectId) setProjectRuns((await getProjectRuns(projectId, signal)).runs);
  }, []);

  const loadBundle = async (runId: string, signal: AbortSignal) =>
    loadViewerBundle(await getViewerManifest(runId, signal), signal);

  const watchRun = useCallback(async (initial: RunRecord, selectedProject?: ProjectManifest) => {
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    const selection = ++selectionRef.current;
    if (selectedProject) setProject(selectedProject);
    setRun(initial); setReadiness(null); setBundle(null); setError(""); setScreen("PROCESSING");
    setUrl(initial.project_id, initial.run_id);
    let lastBundleKey = "";
    try {
      for (;;) {
        const [fresh, ready] = await Promise.all([
          getRun(initial.run_id, controller.signal),
          getRunReadiness(initial.run_id, controller.signal),
        ]);
        if (selection !== selectionRef.current) return;
        setRun(fresh); setReadiness(ready);
        const bundleKey = `${ready.quality_report_ready}:${ready.dense_status}:${fresh.artifacts.length}`;
        if (ready.sparse_preview_ready && bundleKey !== lastBundleKey) {
          try {
            const loaded = await loadBundle(fresh.run_id, controller.signal);
            if (selection !== selectionRef.current) return;
            lastBundleKey = bundleKey; setBundle(loaded); setScreen("WORKSPACE");
          } catch (cause) {
            if (isAbort(cause)) return;
            setError(cause instanceof Error ? cause.message : "Sparse preview could not be opened.");
          }
        }
        if (TERMINAL.has(fresh.status)) { await refreshCatalog(fresh.project_id, controller.signal); return; }
        await new Promise<void>((resolve, reject) => {
          const timer = window.setTimeout(resolve, 1000);
          controller.signal.addEventListener("abort", () => {
            window.clearTimeout(timer); reject(new DOMException("Polling cancelled", "AbortError"));
          }, { once: true });
        });
      }
    } catch (cause) {
      if (!isAbort(cause) && selection === selectionRef.current)
        setError(cause instanceof Error ? cause.message : "Run state could not be refreshed.");
    }
  }, [refreshCatalog]);

  const openExistingRun = useCallback(async (runId: string) => {
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    const selection = ++selectionRef.current;
    setBusy(true); setError("");
    try {
      const existingRun = await getRun(runId, controller.signal);
      if (selection !== selectionRef.current) return;
      const existingProject = await getProject(existingRun.project_id, controller.signal);
      if (selection !== selectionRef.current) return;
      await refreshCatalog(existingProject.project_id, controller.signal);
      if (selection !== selectionRef.current) return;
      setBusy(false);
      void watchRun(existingRun, existingProject);
    } catch (cause) {
      if (!isAbort(cause) && selection === selectionRef.current) { setError(cause instanceof Error ? cause.message : "The saved run could not be opened."); setScreen("SETUP"); }
    } finally { if (selection === selectionRef.current) setBusy(false); }
  }, [refreshCatalog, watchRun]);

  const selectProject = async (projectId: string) => {
    requestRef.current?.abort(); const selection = ++selectionRef.current; setBusy(true); setError("");
    try {
      const [selected, runs] = await Promise.all([getProject(projectId), getProjectRuns(projectId)]);
      if (selection !== selectionRef.current) return;
      setProject(selected); setProjectRuns(runs.runs); setRun(null); setBundle(null); setReadiness(null); setUrl(projectId);
    } catch (cause) { if (selection === selectionRef.current) setError(cause instanceof Error ? cause.message : "Project could not be selected."); }
    finally { if (selection === selectionRef.current) setBusy(false); }
  };

  const reset = () => {
    requestRef.current?.abort(); ++selectionRef.current;
    setScreen("SETUP"); setProject(null); setProjectRuns([]); setRun(null); setReadiness(null); setBundle(null); setError(""); setActionBusy(""); setUrl();
    void refreshCatalog().catch(() => undefined);
  };

  const handleUpload = async (input: UploadInput) => {
    setBusy(true); setError("");
    try {
      const createdProject = await uploadProject(input);
      const createdRun = await startRun(createdProject.project_id, input.config);
      await refreshCatalog(createdProject.project_id); void watchRun(createdRun, createdProject);
    } catch (cause) { if (!isAbort(cause)) setError(cause instanceof Error ? cause.message : "The run could not be started."); }
    finally { setBusy(false); }
  };

  const handleDemo = async () => {
    setBusy(true); setError("");
    try { const created = await createSyntheticDemo(); await refreshCatalog(created.project.project_id); void watchRun(created.run, created.project); }
    catch (cause) { if (!isAbort(cause)) setError(`${cause instanceof Error ? cause.message : "API unavailable."} You can still open the offline UI fixture.`); }
    finally { setBusy(false); }
  };

  const handleOfflineFixture = async () => {
    requestRef.current?.abort(); ++selectionRef.current; setBusy(true); setError("");
    try { setBundle(await loadOfflineFixture()); setProject(null); setRun(null); setReadiness(null); setScreen("WORKSPACE"); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "Offline fixture could not be loaded."); }
    finally { setBusy(false); }
  };

  const runAction = async (name: string, action: () => Promise<RunRecord>) => {
    setActionBusy(name); setError("");
    try { void watchRun(await action(), project ?? undefined); }
    catch (cause) { setError(cause instanceof Error ? cause.message : `${name} request failed.`); }
    finally { setActionBusy(""); }
  };

  const openPreview = async () => {
    if (!run || !readiness?.sparse_preview_ready) return;
    setActionBusy("preview"); setError("");
    try { setBundle(await loadBundle(run.run_id, new AbortController().signal)); setScreen("WORKSPACE"); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "Sparse preview could not be opened."); }
    finally { setActionBusy(""); }
  };

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("fixture") === "1") void handleOfflineFixture();
    else if (params.get("run")) void openExistingRun(params.get("run")!);
    else {
      const projectId = params.get("project") ?? undefined;
      void refreshCatalog(projectId).then(async () => {
        if (projectId) try { setProject(await getProject(projectId)); } catch { setError("Saved project no longer exists."); }
      }).catch((cause) => setError(cause instanceof Error ? cause.message : "Project catalog unavailable."));
    }
    return () => requestRef.current?.abort();
    // One-time URL recovery; selection changes are explicit.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return <div className="app">
    <header className="app-header">
      <button type="button" className="brand" onClick={reset} aria-label="Trace3D home"><span className="brand-mark"><i /><i /><i /></span><span><strong>TRACE</strong><b>3D</b></span></button>
      <div className="header-context"><span>RECONSTRUCTION EVIDENCE WORKSPACE</span>{run && <code>{run.run_id}</code>}</div>
      <div className="system-state"><span className="system-dot" /><span>API OPERATOR</span><time>{new Date().toLocaleDateString(undefined, { day: "2-digit", month: "short", year: "numeric" })}</time></div>
    </header>
    {screen === "SETUP" && <SetupScreen busy={busy} error={error} projects={projects} selectedProject={project} runs={projectRuns} onSelectProject={(id) => void selectProject(id)} onOpenRun={(id) => void openExistingRun(id)} onRefresh={() => void refreshCatalog(project?.project_id)} onUpload={(input) => void handleUpload(input)} onDemo={() => void handleDemo()} onOfflineFixture={() => void handleOfflineFixture()} />}
    {screen === "PROCESSING" && run && <ProgressScreen run={run} readiness={readiness} error={error} actionBusy={actionBusy} onReset={reset} onCancel={() => void runAction("cancel", () => cancelRun(run.run_id))} onResume={() => void runAction("resume", () => resumeRun(run.run_id))} onOpenPreview={() => void openPreview()} />}
    {screen === "WORKSPACE" && bundle && <Workspace bundle={bundle} project={project} run={run} readiness={readiness} actionBusy={actionBusy} globalError={error} onReset={reset} onCancel={run ? () => void runAction("cancel", () => cancelRun(run.run_id)) : undefined} onResume={run ? () => void runAction("resume", () => resumeRun(run.run_id)) : undefined} onRerun={(config: RunConfiguration) => run && void runAction("rerun", () => rerunRun(run.run_id, config))} />}
  </div>;
}
