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
preview can follow the generated audio motion, and `all` preview animates that
head subset while the learned local feature-plane Gaussians remain static
first-frame diagnostics. Full Gaussian animation still needs a more complete
per-frame contract for local Gaussian deformation, feature-channel rendering,
camera sorting, and the upsampler-equivalent color path.

The backend also exports `gaussians.transforms.f32` as one GAGAvatar
`t_transform` 3x4 matrix per frame. The browser validates this artifact but
does not apply it yet; the next step is to reconcile that transform with the
interactive orbit camera without breaking inspection controls.

The shader preview sorts its instanced splats back-to-front on the CPU when the
camera changes, and on each animated head-frame update in `head` preview mode.
This improves ordinary alpha blending for inspection, but it is still a
preview approximation. A production Gaussian renderer should move sorting and
screen-space covariance handling to a purpose-built WebGL/WebGPU path.

The preview shader also projects each Gaussian's scaled 3D axes into view
space and draws a screen-facing ellipse from the resulting 2D covariance. This
is closer to Gaussian Splatting than drawing world-oriented cards, but it still
omits the CUDA rasterizer's exact projection, filtering, and tile pipeline.

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
