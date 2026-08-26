# SIH26158 operator UI

Arnav's React/TypeScript operator workspace consumes the exact viewer manifest declared in
`examples/viewer-manifest.json` and rendered by `GET /api/runs/:runId/viewer-manifest`.

## Run

```bash
cd frontend
npm install
npm run dev
```

Start the API separately from the repository root with `make api`. Vite proxies `/api` to
`http://127.0.0.1:8000`.

## Safe offline fixture

Open `http://127.0.0.1:5173/?fixture=1`. The fixture is prominently marked synthetic and exists only
for deterministic UI/browser verification. It is not real COLMAP or metric evidence.

## Verify

```bash
npm test
npm run build
npm run test:e2e
```

The end-to-end test uses the locally installed Google Chrome channel and does not require the API.

## Live preprocessing

When the handoff field is blank, the backend creates a dataset-specific preview handoff from the
uploaded video and matching SRT/CSV: uniform 2 Hz frames plus normalized telemetry. An external
handoff can still be supplied for reviewed frame scoring, masks, or a custom selection policy.
