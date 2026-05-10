# Realtime Notes

Design notes for the realtime / streaming-friendly extension of ARTalk
inference. Work happens on the `realtime` branch.

## Constraints

The released model processes audio in fixed 4-second / 100-frame chunks.
Multi-scale autoregressive decoding within each chunk depends on the
full chunk being available, so end-to-end latency cannot drop below
~4 seconds without architectural changes to the AR model (smaller
`patch_nums`, causal redesign, etc.). All work in this branch
preserves `assets/ARTalk_wav2vec.pt` bit-for-bit unchanged; latency
reductions below the 4-second floor are deferred to coordinated work
with the original author.

## Phase plan

### Phase 1 — Streaming inference

Refactor `BitwiseARModel.inference`'s chunked audio loop into a stateful
feed/finish API in `app/streaming.py` (`ARTalkStreamer`). The model
itself and the existing one-shot inference path are untouched, so the
released checkpoint and the CLI / Gradio app continue to work
unchanged.

`scripts/check_streaming_parity.py` asserts the streaming output is
bit-exact with the one-shot path. Verified on the released wav2vec
checkpoint: `max abs diff: 0.000e+00`.

### Phase 2 — Causal motion smoother

Replace the post-processing in `ARTAvatarInferEngine.smooth_motion_savgol`
(scipy `savgol_filter` over the full motion sequence — non-causal)
with a streaming-friendly equivalent (`CausalSavgolSmoother` in
`app/streaming.py`).

#### Design decisions

Three approaches were considered:

- **(A) Delayed savgol.** Buffer future motion frames and emit each
  output with `(POSE_WINDOW - 1) // 2 = 4` frames (160 ms at 25 fps)
  of additional latency. Re-running scipy's `savgol_filter` on the
  growing buffer yields output that is **bit-exact** with the
  one-shot path: `mode='interp'` (the scipy default) is deterministic,
  and interior frames depend only on a fixed `window`-sized
  neighborhood that becomes context-independent once both halves of
  the window are in the buffer.
- **(B) Causal IIR / EMA.** One-pole low-pass filter applied to past
  samples only. Zero added latency, but the frequency response
  differs from savgol; output values would not match the one-shot
  reference.
- **(D) No smoothing.** The original smoother is cosmetic; some use
  cases may not need it.

**Choice: (A).** The 160 ms added latency is small relative to the 4-second
chunk floor, and bit-exact output keeps Phase 2 from being a confounding
variable when verifying later phases. (D) remains available as a simple
opt-out at the call site (do not run motion through the smoother).
(B) is recorded for reference; it would be considered together with
retraining work since it would not be bit-exact regardless.

### Phase 3 — Per-frame rendering

`StreamingRenderer` in `app/rendering.py` mirrors the mesh /
GAGAvatar branches of `ARTAvatarInferEngine.rendering` as a per-frame
API: `render_frame(motion_frame) → (3, H, W)` RGB tensor on CPU in
[0, 1] range. Audio/video muxing is no longer the renderer's
responsibility — it moves to the transport layer (Phase 4 WebRTC
sink).

#### Design decisions

- **Single class with mode dispatch**, not separate per-mode classes.
  Mirrors the existing one-shot `rendering()`'s `shape_id`-based
  dispatch and lets config-driven callers (Phase 4) construct
  uniformly.
- **New file `app/rendering.py`**, not a subsection of
  `app/streaming.py`. Streaming-specific motion logic
  (`ARTalkStreamer`, `CausalSavgolSmoother`) stays focused there;
  rendering is a separate concern with different dependencies (FLAME
  / GAGAvatar) and was getting unwieldy to colocate.
- **Existing one-shot `ARTAvatarInferEngine.rendering()` is left
  untouched.** A small amount of per-frame logic is duplicated
  between the two paths in exchange for keeping the upstream-facing
  diff minimal; the original CLI / Gradio app continues to work
  unchanged.
- **No "raw motion params" mode on the renderer.** Clients that
  want to ship motion params instead of rendered video simply skip
  the renderer stage and consume motion frames directly from
  `CausalSavgolSmoother` (or `ARTalkStreamer`). Adding a passthrough
  mode would have been a thin abstraction that doesn't earn its name.
- **Renderer takes loaded modules as constructor args**, not paths.
  Typical callers pass the modules `ARTAvatarInferEngine` already
  loaded; tests can construct the FLAME pieces directly without
  spinning up a full engine.

`scripts/check_streaming_parity.py` adds a Phase 3 mesh-mode parity
check (skipped automatically if `assets/FLAME_with_eye.pt` is not
present). The GAGAvatar path is not covered by automated parity
because of the asset-download cost; verify it with a smoke test if
needed.

#### Parity criterion (mesh)

Phases 1 and 2 produce **bit-exact** output between streaming and
one-shot. Phase 3 mesh rendering does not, and cannot:

- `get_flame_verts` runs FLAME's linear-blend-skinning either over
  the full batch `T` (one-shot) or `T` times over batch `1`
  (streaming). Float32 add non-associativity makes vertex
  coordinates diverge at the ~1e-7 level depending on reduction
  order.
- Those tiny vertex deltas feed a discrete rasterizer, which makes
  binary coverage decisions at silhouette edges. A 1–2 pixel
  silhouette outline can flip individual pixels.

In practice this manifests as `mean abs diff ~ 2e-8` with
`max abs diff ~ 6e-1` on a 13-second clip. The parity test therefore
asserts **mean abs diff < 1e-5 AND <0.1% of pixels diverge by more
than 1e-2**, instead of strict element-wise equality. Streaming
output is "visually identical" to one-shot, not bit-identical, and
that is the right notion of equivalence for this stage.

### Phase 4 — WebRTC transport

`webrtc_app.py` (root) loads the model + FLAME pieces once, builds a
shared-state `ARTalkHandler` (in `app/webrtc.py`) that subclasses
FastRTC's `AsyncAudioVideoStreamHandler`, and serves the FastRTC dev
UI. Each browser connection gets its own per-connection state
(streamer / smoother / renderer / queues / worker task) via
`copy()` + `start_up()`.

Inside each connection, audio frames received by FastRTC are pushed
to an `asyncio.Queue`; a single per-connection worker drains the
queue and offloads each step (`streamer.feed → smoother.feed →
renderer.feed`) to a thread via `asyncio.to_thread`, so streaming
state stays serialized while the event loop remains responsive for
inbound audio. Rendered RGB frames are pushed to a video queue;
`video_emit()` drains it at whatever rate FastRTC's outbound track
calls for, with the queue absorbing the 100-frame-per-4s bursts.

#### Design decisions

- **Library: FastRTC** (Gradio team), chosen over `streamlit-webrtc`
  and raw `aiortc`. The project already depends on Gradio, FastRTC
  is built specifically for AI streaming use cases, and HF Spaces
  deployment is direct.
- **Single per-connection inference worker** rather than spawning a
  task per audio frame: keeps the streamer / smoother / renderer
  state strictly ordered without locks. Per-frame inference is
  offloaded to a thread so the event loop never blocks on GPU work.
- **Video track only**, no outbound audio. Mixing audio at a 4 s
  lag to match the avatar video is non-trivial and is left for a
  future iteration; users can monitor their own mic via system
  sidetone for lip reference if needed.
- **No startup-pad frames.** The first 4 s of a session has no
  outbound video — FastRTC and the browser handle this naturally
  (the video element starts when the first frame arrives).
- **MVP scope**: mesh mode, no style motion, no UI configurability.
  GAGAvatar mode, style selection, TTS input, and audio echo come
  in later iterations.
- **`fastrtc` is not added to `environment.yml`** to keep the
  upstream-facing dep list unchanged; install with `pip install
  fastrtc` after activating the conda env.

#### Running

```bash
pip install fastrtc        # one-time, after activating the env
python webrtc_app.py       # default: cuda, 0.0.0.0:8000
```

Open the FastRTC dev UI in a browser (`http://<host>:8000/`), grant
microphone permission, start the WebRTC session, speak, and the
avatar starts moving ~4 seconds later.

## Future (deferred / out of branch scope)

- **Architectural latency reduction below the 4-second floor.** Smaller
  `patch_nums`, causal AR redesign, etc. Requires retraining and
  coordination with the original author.
- **Client-side rendering.** Push raw 106-dim motion params over the
  wire instead of rendered video and implement FLAME / GAGAvatar
  equivalents in the browser (Three.js + Gaussian splat). Drastically
  reduces server load and bandwidth, and decouples the rendering
  pipeline from the inference server.
