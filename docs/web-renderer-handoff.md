# Web Renderer Handoff

Last updated: 2026-06-07.

## Current State

The web renderer work lives on branch `web-based-renderer`, tracking
`origin/web-based-renderer`.

Latest known head:

```text
90c2da6 feat(web): add server-side GAGAvatar video mode
```

Open PR:

```text
https://github.com/whitphx/ARTalk/pull/1
```

GitHub reported no checks configured for this PR branch.

Recent commits:

- `9d6e3ab feat(web): add mesh renderer API`
- `03d9919 feat(frontend): add Three.js web renderer`
- `aceb261 docs(web): document macOS web setup`
- `2e3b291 refactor(frontend): split renderer modules`
- `312d46b feat(web): add GAGAvatar avatar registration`
- `90c2da6 feat(web): add server-side GAGAvatar video mode`

Implemented capabilities:

- FastAPI backend for ARTalk motion generation.
- React/Vite frontend for interactive generation.
- Browser-side Three.js FLAME mesh playback.
- Audio and text input.
- Style, device, avatar, and output-mode selectors.
- Built-in GAGAvatar tracked identities drive mesh geometry via `shapecode`.
- Uploaded face registration through GAGAvatar_track.
- Uploaded avatar preview images.
- Copyable, scrollable frontend error panel.
- Optional server-side GAGAvatar colored-video mode.

## Design

The web demo is split into a FastAPI backend and a React/Vite frontend.

Backend responsibilities:

- Accept audio or text input.
- Run ARTalk motion inference.
- Write generated motion, audio, mesh vertices, and mesh faces to
  `render_results/web_jobs/<job_id>/`.
- Track uploaded face images through GAGAvatar_track in a separate Python
  environment.
- Register neutral, built-in, and uploaded avatars as selectable identities.
- Render optional server-side GAGAvatar colored MP4 output when CUDA gaussian
  rasterization is available.

Frontend responsibilities:

- Collect input audio/text, style, device, avatar, and output mode.
- Poll background jobs.
- Render mesh output in the browser with Three.js.
- Display server-rendered colored MP4 output when `videoUrl` is present.
- Keep error messages scrollable and copyable.

Key implementation files:

- `web_app.py`: FastAPI API, job routing, static frontend serving.
- `app/web_inference.py`: ARTalk web inference and mesh artifact writing.
- `app/avatar_registry.py`: neutral, built-in, and uploaded avatar registry.
- `app/gagavatar_tracking.py`: uploaded face tracking subprocess wrapper.
- `app/gagavatar_video.py`: CUDA-only server-side colored MP4 renderer.
- `frontend/src/App.tsx`: controls and job polling.
- `frontend/src/components/Renderer.tsx`: Three.js mesh preview or MP4 preview.

## Output Modes

`mesh` is the default mode.

- Runs on macOS.
- Uses ARTalk motion and FLAME vertices.
- Uses selected avatar `shapecode` for geometry.
- Sends `vertices.f32`, `faces.i32`, and `audio.wav` to the browser.
- Frontend renders the animated mesh with Three.js.

`gagavatar` is the server-side colored-video mode.

- Requires CUDA and `diff_gaussian_rasterization_32d`.
- Uses ARTalk motion and selected GAGAvatar tracked identity.
- Writes `gagavatar.mp4` under the job directory.
- Frontend displays the MP4 instead of the Three.js mesh renderer.
- On macOS CPU/MPS, the backend rejects the request before job creation with a
  CUDA requirement message.

## Avatar Model

Avatar IDs use these formats:

- `mesh`: neutral FLAME mesh.
- `gagavatar:<key>`: built-in identity from `assets/GAGAvatar/tracked.pt`.
- `uploaded:<avatar_id>`: user-registered identity from
  `render_results/web_avatars/<avatar_id>/tracked.pt`.

Uploaded avatar registration writes:

- `upload.<ext>`: original uploaded image.
- `input.<ext>`: copied tracker input.
- `tracked.pt`: tracked GAGAvatar identity payload.
- `preview.jpg`: tracking preview.
- `metadata.json`: local avatar metadata.
- `state.json`: async job state.

## API Contract

`GET /api/config`

Returns styles, avatars, languages, render modes, and defaults.

`POST /api/avatar-jobs`

Form fields:

- `device`: defaults to `auto`.
- `image_file`: JPG or PNG.

Creates an uploaded avatar tracking job.

`GET /api/avatar-jobs/{avatar_id}`

Returns avatar tracking state.

`GET /api/avatar-jobs/{avatar_id}/preview.jpg`

Returns the uploaded avatar preview after tracking completes.

`POST /api/jobs`

Form fields:

- `input_type`: `audio` or `text`.
- `audio_file`: required for audio input.
- `text`, `text_language`: required for text input.
- `style_id`
- `clip_length`
- `device`
- `avatar_id`
- `render_mode`: `mesh` or `gagavatar`.

Creates an ARTalk generation job.

`GET /api/jobs/{job_id}`

Returns generation job state.

`GET /api/jobs/{job_id}/metadata`

Returns absolute URLs for generated artifacts. `videoUrl` is `null` for mesh
mode and points to `gagavatar.mp4` for colored-video mode.

`GET /api/jobs/{job_id}/{name}`

Allows:

- `vertices.f32`
- `faces.i32`
- `audio.wav`
- `motions.pt`
- `gagavatar.mp4`

## Fresh Setup

Clone and checkout:

```bash
git clone https://github.com/whitphx/ARTalk.git
cd ARTalk
git checkout web-based-renderer
git submodule update --init --recursive
```

Verify required assets:

```bash
ls assets/ARTalk_wav2vec.pt assets/config.json assets/FLAME_with_eye.pt
ls assets/GAGAvatar/GAGAvatar.pt assets/GAGAvatar/tracked.pt
```

Create the main env and install frontend deps:

```bash
micromamba create -f environment-web.yml
micromamba run -n artalk-web scripts/check_web_backend_env.py
cd frontend
pnpm install
cd ..
```

Run the backend:

```bash
micromamba run -n artalk-web scripts/run_web_backend.sh
```

The launcher prepends `$CONDA_PREFIX/lib` to `LD_LIBRARY_PATH` so Linux uses
the FFmpeg and C++ runtime libraries from the `artalk-web` environment before
older system copies. Override the backend URL used by Vite with
`ARTALK_API_TARGET` if you do not run the API on `http://127.0.0.1:8961`.

Run the frontend:

```bash
cd frontend
pnpm dev --host 127.0.0.1 --port 5173
```

Open:

```text
http://127.0.0.1:5173/
```

## Uploaded Face Registration Setup

The web renderer uses the repo-local `GAGAvatar/` submodule by default. Set
`GAGAVATAR_REPO` only when you intentionally want to use another checkout.

Create the tracking env:

```bash
micromamba create -f environment-gagavatar-track.yml
```

Download GAGAvatar_track resources:

```bash
cd GAGAvatar/core/libs/GAGAvatar_track
curl -L -o track_resources.tar \
  https://huggingface.co/xg-chu/GAGAvatar_track/resolve/main/track_resources.tar
tar -xf track_resources.tar
rm track_resources.tar
cd ../../../..
```

Confirm key resources:

```bash
ls assets/flame/FLAME_with_eye.pt
ls assets/emica/EMICA-CVT_flame2020_notexture.pt
ls assets/matting/stylematte_synth.pt
```

Check the tracker environment:

```bash
micromamba run -n artalk-web scripts/check_gagavatar_tracker_env.py \
  --python "$HOME/.local/share/mamba/envs/gagavatar-track/bin/python"
```

Run backend with the tracker Python configured:

```bash
GAGAVATAR_PYTHON="$HOME/.local/share/mamba/envs/gagavatar-track/bin/python" \
micromamba run -n artalk-web scripts/run_web_backend.sh
```

## CUDA Colored Video Setup

Install the GAGAvatar Gaussian rasterizer into the backend env:

```bash
CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=6.0 \
  micromamba run -n artalk-web scripts/install_gagavatar_rasterizer.sh
```

Use the CUDA architecture for the target GPU. For example, Tesla P100 is `6.0`.

Run the full backend preflight:

```bash
micromamba run -n artalk-web scripts/check_web_backend_env.py --full
```

## Validation

Backend import check:

```bash
micromamba run -n artalk-web python -m py_compile \
  web_app.py app/web_inference.py app/avatar_registry.py \
  app/gagavatar_tracking.py app/gagavatar_video.py
```

Frontend build:

```bash
cd frontend
pnpm build
```

Config endpoint:

```bash
curl -sS http://127.0.0.1:8961/api/config
```

Expected: response includes `renderModes` with `mesh` and `gagavatar`.

Mesh job:

```bash
curl -sS -X POST http://127.0.0.1:8961/api/jobs \
  -F input_type=audio \
  -F audio_file=@demo/eng1.wav \
  -F clip_length=100 \
  -F render_mode=mesh \
  -F avatar_id=mesh
```

Poll the returned job ID and fetch metadata:

```bash
curl -sS http://127.0.0.1:8961/api/jobs/<job_id>
curl -sS http://127.0.0.1:8961/api/jobs/<job_id>/metadata
```

Expected: `status=complete`, `renderMode=mesh`, and `videoUrl=null`.

Colored-mode macOS preflight:

```bash
curl -sS -i -X POST http://127.0.0.1:8961/api/jobs \
  -F input_type=text \
  -F text=hello \
  -F render_mode=gagavatar \
  -F avatar_id=gagavatar:11.jpg
```

Expected on macOS: HTTP 400 with a CUDA requirement message.

Uploaded avatar registration:

1. Start backend with `GAGAVATAR_REPO` and `GAGAVATAR_PYTHON`.
2. Upload a JPG/PNG through the frontend.
3. Wait for avatar state to become `complete`.
4. Confirm `/api/config` includes `uploaded:<avatar_id>`.
5. Generate a mesh job with that uploaded avatar.

CUDA colored-video validation:

1. Use a backend environment with CUDA and `diff_gaussian_rasterization_32d`.
2. Submit `render_mode=gagavatar` with a built-in or uploaded avatar.
3. Confirm metadata has `renderMode=gagavatar` and non-null `videoUrl`.
4. Confirm frontend displays the generated MP4.

## Known Local Verification

Before this handoff, these checks passed locally:

- Python compile check for backend modules.
- `pnpm build`.
- `GET /api/config`.
- Short mesh job using `demo/eng1.wav`.
- Mesh metadata check.
- macOS colored-mode preflight returning HTTP 400 with CUDA requirement.
- Browser smoke test showing the `Output` selector.

Backend command used locally:

```bash
GAGAVATAR_REPO=/Users/whitphx/ghq/github.com/xg-chu/GAGAvatar \
GAGAVATAR_PYTHON=/Users/whitphx/.local/share/mamba/envs/gagavatar-track/bin/python \
/Users/whitphx/.local/share/mamba/envs/artalk-web/bin/python \
  -m uvicorn web_app:app --host 0.0.0.0 --port 8961
```

Frontend command used locally:

```bash
cd frontend
pnpm dev --host 127.0.0.1 --port 5173
```

## Constraints And Guardrails

- Keep `mesh` as the default output mode.
- Keep the macOS-compatible path mesh-only.
- Do not import GAGAvatar rendering dependencies on startup unless colored
  rendering is requested.
- Preserve the early failure for colored rendering on non-CUDA machines.
- Keep GAGAvatar_track in a separate subprocess environment.
- TorchDynamo is disabled in the tracker subprocess to avoid macOS Inductor
  OpenMP compilation failures.
- The uploaded-avatar tracker may emit noisy Objective-C duplicate-class
  warnings from `av` and `cv2`; those were observed as non-blocking.
- Generated job and uploaded avatar directories are runtime artifacts, not
  source handoff content.
- GAGAvatar_track is CC BY-NC 4.0; production or commercial use needs separate
  license review.

## Next Work

- Validate `render_mode=gagavatar` on an actual CUDA host.
- Package or document installation for `diff_gaussian_rasterization_32d`.
- Add automated tests around `GET /api/config`, mesh metadata, and non-CUDA
  colored-mode preflight.
- Consider moving long-running generation/tracking from FastAPI background
  tasks to a durable job queue if this becomes a public web service.
- Add cleanup policy for `render_results/web_jobs` and
  `render_results/web_avatars`.
