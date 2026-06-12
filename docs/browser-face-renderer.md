# Browser Face Renderer Plan

The browser renderer should evolve in stages. The current implementation keeps
the FLAME mesh renderer as the interactive baseline and uses server-rendered
MP4 output for GAGAvatar colored video.

## Current Browser Renderer

- `Renderer.tsx` owns the stage shell, loading overlay, transport, and renderer
  selection.
- `MeshFaceRenderer.tsx` renders ARTalk mesh artifacts from `vertices.f32`,
  `faces.i32`, and `audio.wav`.
- `VideoRenderer.tsx` displays server-side GAGAvatar MP4 output.

This split keeps the mesh preview and server-video fallback independent while
we improve browser rendering quality.

## Mesh Renderer Improvements

The mesh path is still the first browser-side target because it uses artifacts
already produced by the backend and gives a reliable fallback for every
browser. Near-term improvements should stay focused on:

- face-oriented material and lighting presets
- stable buffer updates with `DynamicDrawUsage`
- camera framing that centers the generated head consistently
- playback controls that keep audio and frame selection synchronized
- renderer controls for material mode, wireframe, and camera reset

The current `region` material mode consumes `regions.u8` when generated job
metadata provides it. That file is a per-vertex `uint8` label buffer using the
metadata `regionLabels` map. The backend source is currently
`mediapipe-landmark-adjacency-v1`: it seeds eye and mouth regions from the
Mediapipe landmark embeddings stored in the FLAME checkpoint, then grows those
seeds over mesh adjacency. The FLAME checkpoint does not include a UV map or
semantic face-region masks, so these labels are topology-guided visualization
regions rather than production segmentation.

The important contract is now in place: a future backend can replace
`regions.u8` with stronger semantic labels without changing the browser
renderer. Older jobs without `regionLabelsUrl` still fall back to the
browser-side normalized-position heuristic.

The default `skin` material also consumes the region labels, but with a subdued
palette intended for presentation rather than inspection. The explicit `region`
material keeps the higher-contrast diagnostic colors.

Playback controls are owned by the renderer shell while the hidden audio
element remains the timing source for mesh frame selection. Scrubbing updates
the audio clock directly, so paused and playing states use the same frame-sync
path.

The mesh renderer runs a continuous animation loop only while audio is playing
or orbit controls are damping. In paused states it redraws on explicit events
such as seek, resize, camera reset, or material changes.

During playback, the mesh renderer samples the audio clock at display refresh
and linearly interpolates between adjacent generated vertex frames. The
backend artifacts remain 25 fps, but browser playback avoids visibly stepping
between those frames.

## Future Gaussian Splatting Extension

GAGAvatar's server-side colored renderer is based on a learned Gaussian avatar,
so visual parity in the browser eventually requires browser-side Gaussian
Splatting rather than only a better FLAME mesh material.

The extension should be handled as a separate renderer mode, not by replacing
the mesh fallback:

- `mesh`: current browser FLAME mesh preview
- `server-video`: server-rendered GAGAvatar MP4
- `browser-gaussian`: future browser-side Gaussian renderer

The experimental `browser-gaussian` path starts by exporting a first-frame
GAGAvatar Gaussian snapshot and displaying it with a Three.js shader splat
preview. The preview consumes position, color, opacity, scale, and rotation
buffers and draws instanced quad splats. It is still an approximation of the
CUDA rasterizer rather than a true screen-space covariance implementation.

The diagnostic preview can show only the first 5,023 FLAME/head Gaussians, only
the learned local feature-plane Gaussians, or both. GAGAvatar's browser path
cannot match server video until it also handles 32-channel feature rendering and
the neural upsampler output path.

The spike now exports per-frame positions for the first 5,023 head Gaussians.
That keeps the lightest useful animated path in the browser: the `head`
preview can follow the generated audio motion with interpolated centers between
generated frames, and `all` preview animates that head subset while the learned
local feature-plane Gaussians remain static first-frame diagnostics. Full
Gaussian animation still needs a more complete per-frame contract for local
Gaussian deformation, feature-channel rendering, camera sorting, and the
upsampler-equivalent color path.

Because the current `all` preview combines animated head Gaussians with
first-frame static local feature-plane Gaussians, the browser slightly boosts
the head subset and reduces feature-plane opacity in that composite mode. This
keeps mouth and expression motion inspectable while the browser path lacks the
server renderer's 32-channel rasterization and neural upsampler.

The backend also exports `gaussians.transforms.f32` as one GAGAvatar
`t_transform` 3x4 matrix per frame. The browser validates this artifact but
keeps the interactive `orbit` view as the default inspection mode. The
experimental `GAGAvatar` view applies the per-frame transform in the shader so
the browser preview can be compared against the server renderer's camera
convention.

In `GAGAvatar` view, the browser renderer applies two internal transform
conventions: local feature-plane Gaussians use the native exported view matrix,
while the animated head subset uses the X-mirrored view matrix. Both matrices
preserve the first frame and invert the per-frame transform delta. This matches
the server-rendered reference most closely in current visual checks and avoids
exposing the earlier diagnostic X controls in the UI.

This is still an experimental browser convention. Before treating it as a
stable artifact contract, verify whether the inverse transform delta belongs in
the exported `gaussians.transforms.f32` data or only in the browser's mapping
from GAGAvatar's rasterizer convention to Three.js view space.

The shader preview sorts its instanced splats back-to-front on the CPU when the
camera changes, and on each animated head-frame update in `head` preview mode.
This improves ordinary alpha blending for inspection, but it is still a
preview approximation. A production Gaussian renderer should move sorting and
screen-space covariance handling to a purpose-built WebGL/WebGPU path.

The preview shader also projects each Gaussian's scaled 3D axes into view
space and draws a screen-facing ellipse from the resulting 2D covariance. This
is closer to Gaussian Splatting than drawing world-oriented cards, but it still
omits the CUDA rasterizer's exact projection, filtering, and tile pipeline.

Color is still diagnostic. The browser splat preview maps the first three of
GAGAvatar's 32 learned feature channels to RGB, while the server renderer
rasterizes all 32 channels and runs the neural upsampler. Browser Gaussian jobs
therefore export `gaussians.reference.mp4`, a synced server-side GAGAvatar
reference video, so color and camera experiments can be compared against the
real target while the browser color path is still approximate.

Browser Gaussian jobs also export a capped sampled sequence of separate
`gaussians.upsampler_input_*.f16` files, plus the compatibility first-frame
file `gaussians.upsampler_input_first.f16`. Each sample is the real 32-channel
`512 x 512` rasterizer output consumed by GAGAvatar's `StyleUNet` upsampler.
This is a feasibility artifact for the future browser upsampler path: it lets
us validate ONNX inference and playback synchronization against exact server
tensors without first implementing browser-side 32-channel Gaussian
rasterization. Exporting every frame in this raw format is not practical for
normal playback because one float16 frame is about 16 MB.

The default spike export samples 32 frames, which is about 512 MB of raw
upsampler input per generated job. The frames are split into individual files
so the browser can fetch and run the ONNX upsampler incrementally instead of
waiting for one large transfer. Set `ARTALK_UPSAMPLER_PREVIEW_FRAMES` on the
backend process to lower or raise that validation budget. Use
`ARTALK_UPSAMPLER_PREVIEW_FRAMES=all` or
`ARTALK_UPSAMPLER_PREVIEW_STRIDE=1` only for short clips, because every exported
frame costs about 16 MB of raw tensor data.

The helper script `scripts/export_gagavatar_upsampler_onnx.py` exports the
trained `StyleUNet` to `frontend/public/models/gagavatar_upsampler.onnx`.
Export currently expects the backend Python environment to have `onnx`
installed, and browser inference still needs a runtime such as
`onnxruntime-web`. The first browser validation target should run this ONNX
model on the sampled upsampler tensor sequence, compare those frames with
`gaussians.reference.mp4`, and use the first-frame file as a compatibility
fallback for older jobs.

Before implementing `browser-gaussian`, inspect and define the data contract
for `GAGAvatar.forward_expression(...)`:

- canonical Gaussian attributes
- per-frame expression/deformation data
- color or spherical harmonic representation
- opacity, scale, and rotation representation
- camera parameters and coordinate conventions

The backend should then export compact binary artifacts for the browser. Start
with one static Gaussian frame before attempting animation. After static
rendering is reliable, add per-frame deltas or per-frame Gaussian buffers and
measure memory/bandwidth costs.

Candidate browser implementations include Three.js-compatible Gaussian Splat
renderers such as `GaussianSplats3D`, but the integration must account for
GAGAvatar's dynamic per-frame avatar deformation. Keep server-side MP4 output
as the fallback until browser Gaussian rendering is stable across target GPUs.
