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

## Future Gaussian Splatting Extension

GAGAvatar's server-side colored renderer is based on a learned Gaussian avatar,
so visual parity in the browser eventually requires browser-side Gaussian
Splatting rather than only a better FLAME mesh material.

The extension should be handled as a separate renderer mode, not by replacing
the mesh fallback:

- `mesh`: current browser FLAME mesh preview
- `server-video`: server-rendered GAGAvatar MP4
- `browser-gaussian`: future browser-side Gaussian renderer

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
