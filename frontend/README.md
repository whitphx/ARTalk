# ARTalk Web Renderer Frontend

React + Vite frontend for the macOS-compatible ARTalk demo. The Python API
generates FLAME mesh animation buffers; this app renders them in the browser
with Three.js and syncs playback to the returned audio.

```bash
pnpm install
pnpm dev
```

The Vite dev server proxies `/api` to `http://127.0.0.1:8961`.
