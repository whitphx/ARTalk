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

Refactor `ARTAvatarInferEngine.rendering` from "collect all frames →
write_video" into a per-frame iterator. The output channel will be
abstracted so callers can switch between rendered RGB frames and raw
106-dim motion params, leaving the door open for future client-side
rendering (see Future).

### Phase 4 — WebRTC transport

Wire the streaming pipeline to a browser-side audio source and video
sink. Candidate libraries are FastRTC (Gradio team), streamlit-webrtc,
and raw aiortc. Selection is deferred to the start of Phase 4.

## Future (deferred / out of branch scope)

- **Architectural latency reduction below the 4-second floor.** Smaller
  `patch_nums`, causal AR redesign, etc. Requires retraining and
  coordination with the original author.
- **Client-side rendering.** Push raw 106-dim motion params over the
  wire instead of rendered video and implement FLAME / GAGAvatar
  equivalents in the browser (Three.js + Gaussian splat). Drastically
  reduces server load and bandwidth, and decouples the rendering
  pipeline from the inference server.
