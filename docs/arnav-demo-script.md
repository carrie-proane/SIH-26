# Arnav three-minute operator demo

## Before the timer

1. Start the local API with `make api`.
2. Start the frontend with `make ui`.
3. Confirm the real primary run has completed and its viewer manifest opens.
4. Keep `http://127.0.0.1:5173/?fixture=1` open only as the clearly labelled UI fallback.

## Judge route

**0:00–0:30 — trustworthy input**

- Show video + telemetry selection and the required preprocessing-handoff path.
- State that uploads remain local and are checksummed into an immutable manifest.
- Do not call the synthetic fixture a reconstruction.

**0:30–1:00 — live process evidence**

- Start the prepared real run or open its completed record.
- Point to exact ingest → preprocess → reconstruct → report states and the actionable failure field.
- Mention that failed runs retain completed artifacts rather than showing a false success screen.

**1:00–1:45 — reconstruction and provenance**

- Orbit the PLY, show the camera path, then select two source-frame entries.
- Toggle high, medium and low observed confidence layers.
- Show the source-frame unavailable state only if the backend did not declare image URLs; do not
  pretend the procedural placeholder is source imagery.

**1:45–2:25 — measurement trust behavior**

- Enable Measure and select two observed points.
- Show the confidence-based allow/caution/confirm status.
- Show the independently known reference and its percent error in the trust report.
- If the run is synthetic, explicitly say the numbers are illustrative and cannot validate scale.

**2:25–2:50 — AI boundary**

- If real Depth Anything output exists, show it only in the source panel with
  `AI_ASSISTED_NOT_MEASURABLE` visible.
- Show that it never becomes cloud geometry and cannot be used by the distance tool.
- If no real overlay exists, show the disabled toggle and say the experiment is blocked on inputs.

**2:50–3:00 — close**

- Open warnings/limitations.
- State the honest scope: controlled 30–60 second pass, observed surfaces only, local/offline, not
  survey grade.

## Fallback

The `?fixture=1` route proves that the browser workspace can load offline. Its banner says
`UI / orchestration fixture`; it is not reconstruction, registration, scale, or ML evidence.
