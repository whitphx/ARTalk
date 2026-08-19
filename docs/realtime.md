# Realtime streaming notes

Design notes for the streaming-friendly extension of ARTalk inference
(roadmap: [#30](https://github.com/xg-chu/ARTalk/issues/30)).

## Constraints

The released model processes audio in fixed 4-second / 100-frame chunks.
Multi-scale autoregressive decoding within each chunk depends on the
full chunk being available, so end-to-end latency cannot drop below
~4 seconds without architectural changes to the AR model (smaller
`patch_nums`, causal redesign, etc.). All streaming work preserves the
released checkpoints bit-for-bit unchanged; latency reductions below
the 4-second floor would require coordinated retraining work.

## Streaming inference — `ARTalkStreamer`

Refactors `BitwiseARModel.inference`'s chunked audio loop into a
stateful feed/finish API in `artalk/streaming.py`. The model itself and
the existing one-shot inference path are untouched, so the released
checkpoint and the CLI / Gradio app continue to work unchanged.

`scripts/check_streaming_parity.py` asserts the streaming output is
bit-exact with the one-shot path. Verified on the released wav2vec
checkpoint: `max abs diff: 0.000e+00`.

## Causal motion smoother — `CausalSavgolSmoother`

Streaming-friendly equivalent of the post-processing in
`ARTAvatarInferEngine.smooth_motion_savgol` (scipy `savgol_filter` over
the full motion sequence — non-causal).

### Design decisions

Three approaches were considered:

- **Delayed savgol.** Buffer future motion frames and emit each
  output with `(POSE_WINDOW - 1) // 2 = 4` frames (160 ms at 25 fps)
  of additional latency. Re-running scipy's `savgol_filter` on the
  growing buffer yields output that is **bit-exact** with the
  one-shot path: `mode='interp'` (the scipy default) is deterministic,
  and interior frames depend only on a fixed `window`-sized
  neighborhood that becomes context-independent once both halves of
  the window are in the buffer.
- **Causal IIR / EMA.** One-pole low-pass filter applied to past
  samples only. Zero added latency, but the frequency response
  differs from savgol; output values would not match the one-shot
  reference.
- **No smoothing.** The original smoother is cosmetic; some use
  cases may not need it.

**Choice: delayed savgol.** The 160 ms added latency is small relative
to the 4-second chunk floor, and bit-exact output keeps the smoother
from being a confounding variable when verifying the later streaming
stages. "No smoothing" remains available as a simple opt-out at the
call site (do not run motion through the smoother). The causal IIR
option is recorded for reference; it would be considered together with
retraining work since it would not be bit-exact regardless.

Other engine post-processing (eye-channel zeroing, `fix_pose`,
`clip_length` truncation) is not streaming-aware and stays at the
engine layer.

## Streaming rendering — `StreamingRenderer`

`StreamingRenderer` in `artalk/rendering.py` mirrors the mesh /
GAGAvatar branches of `ARTAvatarInferEngine.rendering` as a streaming
API: `render_frame(motion_frame) → (3, H, W)` RGB tensor on CPU in
[0, 1] range, plus `render_batch(motion_frames)` for rendering several
frames per model invocation. Audio/video muxing is no longer the
renderer's responsibility — it moves to the caller's transport layer.

### Design decisions

- **Single class with mode dispatch**, not separate per-mode classes.
  Mirrors the existing one-shot `rendering()`'s `shape_id`-based
  dispatch and lets config-driven callers construct uniformly.
- **New file `artalk/rendering.py`**, not a subsection of
  `artalk/streaming.py`. Streaming-specific motion logic
  (`ARTalkStreamer`, `CausalSavgolSmoother`) stays focused there;
  rendering is a separate concern with different dependencies (FLAME
  / GAGAvatar).
- **Existing one-shot `ARTAvatarInferEngine.rendering()` is left
  untouched.** A small amount of per-frame logic is duplicated
  between the two paths in exchange for keeping the diff minimal; the
  original CLI / Gradio app continues to work unchanged.
- **No "raw motion params" mode on the renderer.** Clients that
  want to ship motion params instead of rendered video simply skip
  the renderer stage and consume motion frames directly from
  `CausalSavgolSmoother` (or `ARTalkStreamer`). Adding a passthrough
  mode would have been a thin abstraction that doesn't earn its name.
- **Renderer takes loaded modules as constructor args**, not paths.
  Typical callers pass the modules `ARTAvatarInferEngine` already
  loaded; tests can construct the FLAME pieces directly without
  spinning up a full engine.

`scripts/check_streaming_parity.py` adds mesh-mode parity checks for
both the per-frame and batched paths (skipped automatically if
`assets/FLAME_with_eye.pt` is not present). The GAGAvatar path is not
covered by automated parity because of the asset-download cost; verify
it with a smoke test if needed.

### Parity criterion (mesh)

The streaming inference and smoother stages produce **bit-exact**
output between streaming and one-shot. Mesh rendering does not, and
cannot:

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

## Planned follow-ups

Per the [roadmap](https://github.com/xg-chu/ARTalk/issues/30): a
realtime pipeline that wires
`ARTalkStreamer → CausalSavgolSmoother → StreamingRenderer` behind
audio-clock-synchronized output callbacks. A WebRTC application layer
built on these pieces lives in
[artalk-streamlit-realtime](https://github.com/whitphx/artalk-streamlit-realtime).

## Future (deferred)

- **Architectural latency reduction below the 4-second floor.** Smaller
  `patch_nums`, causal AR redesign, etc. Requires retraining and
  coordination with the original author.
- **Client-side rendering.** Push raw 106-dim motion params over the
  wire instead of rendered video and implement FLAME / GAGAvatar
  equivalents in the browser (Three.js + Gaussian splat). Drastically
  reduces server load and bandwidth, and decouples the rendering
  pipeline from the inference server.
