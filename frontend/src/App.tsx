import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent, FormEvent } from 'react'
import { Loader2, Mic2, Play, Radio, Upload, Waves } from 'lucide-react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import './App.css'

type InputMode = 'audio' | 'text'

type Config = {
  styles: string[]
  languages: string[]
  defaultStyle: string
}

type JobState = {
  id: string
  status: 'queued' | 'running' | 'complete' | 'failed'
  stage: string
  metadata?: string
  frameCount?: number
  error?: string
}

type AnimationMetadata = {
  fps: number
  sampleRate: number
  frameCount: number
  vertexCount: number
  faceCount: number
  verticesUrl: string
  facesUrl: string
  audioUrl: string
  motionsUrl: string
}

const sleep = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms))

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `${res.status} ${res.statusText}`)
  }
  return res.json()
}

function Renderer({ metadata }: { metadata: AnimationMetadata | null }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [loadState, setLoadState] = useState('Waiting for animation data')

  useEffect(() => {
    if (!metadata || !canvasRef.current) return

    const animation = metadata
    let disposed = false
    let animationId = 0
    let resizeObserver: ResizeObserver | null = null
    const canvas = canvasRef.current
    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
    })
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(28, 1, 0.01, 100)
    const controls = new OrbitControls(camera, canvas)
    const group = new THREE.Group()
    let activeFrame = -1

    controls.enableDamping = true
    controls.enablePan = false
    controls.minDistance = 0.35
    controls.maxDistance = 3.5
    scene.add(group)
    scene.add(new THREE.HemisphereLight(0xe8f3ff, 0x39402e, 2.2))

    const key = new THREE.DirectionalLight(0xffffff, 2.8)
    key.position.set(1.2, 1.6, 2.4)
    scene.add(key)

    const rim = new THREE.DirectionalLight(0x73f4c8, 1.5)
    rim.position.set(-1.4, 0.3, 1.8)
    scene.add(rim)

    async function load() {
      setLoadState('Loading mesh buffers')
      const [vertexBuffer, faceBuffer] = await Promise.all([
        fetch(animation.verticesUrl).then((res) => res.arrayBuffer()),
        fetch(animation.facesUrl).then((res) => res.arrayBuffer()),
      ])
      if (disposed) return

      const allVertices = new Float32Array(vertexBuffer)
      const faces = new Uint32Array(faceBuffer)
      const frameSize = animation.vertexCount * 3
      const positions = new Float32Array(frameSize)
      positions.set(allVertices.subarray(0, frameSize))

      const geometry = new THREE.BufferGeometry()
      geometry.setIndex(new THREE.BufferAttribute(faces, 1))
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
      geometry.computeVertexNormals()
      geometry.computeBoundingSphere()

      const material = new THREE.MeshStandardMaterial({
        color: 0x8eb3f7,
        roughness: 0.42,
        metalness: 0.05,
      })
      const mesh = new THREE.Mesh(geometry, material)
      group.add(mesh)

      const sphere = geometry.boundingSphere
      if (sphere) {
        group.position.set(-sphere.center.x, -sphere.center.y, -sphere.center.z)
        camera.position.set(0, 0.02, Math.max(sphere.radius * 4.5, 0.7))
        controls.target.set(0, 0.01, 0)
        controls.update()
      }

      function resize() {
        const parent = canvas.parentElement
        if (!parent) return
        const { width, height } = parent.getBoundingClientRect()
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
        renderer.setSize(width, height, false)
        camera.aspect = width / Math.max(height, 1)
        camera.updateProjectionMatrix()
      }

      resizeObserver = new ResizeObserver(resize)
      resizeObserver.observe(canvas.parentElement ?? canvas)
      resize()

      function renderLoop() {
        if (disposed) return
        const audio = audioRef.current
        const frame = audio
          ? Math.min(animation.frameCount - 1, Math.floor(audio.currentTime * animation.fps))
          : 0
        if (frame !== activeFrame) {
          activeFrame = frame
          const offset = frame * frameSize
          positions.set(allVertices.subarray(offset, offset + frameSize))
          geometry.attributes.position.needsUpdate = true
          geometry.computeVertexNormals()
        }
        controls.update()
        renderer.render(scene, camera)
        animationId = window.requestAnimationFrame(renderLoop)
      }

      setLoadState('Ready')
      renderLoop()
    }

    load().catch((error: unknown) => {
      if (!disposed) setLoadState(error instanceof Error ? error.message : String(error))
    })

    return () => {
      disposed = true
      window.cancelAnimationFrame(animationId)
      resizeObserver?.disconnect()
      renderer.dispose()
      controls.dispose()
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh) {
          object.geometry.dispose()
          const materials = Array.isArray(object.material) ? object.material : [object.material]
          materials.forEach((material) => material.dispose())
        }
      })
    }
  }, [metadata])

  return (
    <section className="stage" aria-label="Generated avatar preview">
      <div className="viewport">
        <canvas ref={canvasRef} aria-label="3D avatar renderer" />
        {!metadata && (
          <div className="empty-state">
            <Radio aria-hidden="true" />
            <span>No animation loaded.</span>
          </div>
        )}
        {metadata && loadState !== 'Ready' && (
          <div className="loading-state">
            <Loader2 aria-hidden="true" />
            <span>{loadState}</span>
          </div>
        )}
      </div>
      <div className="transport">
        <audio ref={audioRef} src={metadata?.audioUrl} controls />
        <div className="readout" aria-live="polite">
          {metadata
            ? `${metadata.frameCount} frames · ${metadata.vertexCount} vertices · ${metadata.fps} fps`
            : 'No render loaded'}
        </div>
      </div>
    </section>
  )
}

function App() {
  const [config, setConfig] = useState<Config>({
    styles: ['default'],
    languages: ['English'],
    defaultStyle: 'default',
  })
  const [mode, setMode] = useState<InputMode>('audio')
  const [audioFile, setAudioFile] = useState<File | null>(null)
  const [text, setText] = useState('')
  const [language, setLanguage] = useState('English')
  const [style, setStyle] = useState('default')
  const [clipLength, setClipLength] = useState(300)
  const [device, setDevice] = useState('auto')
  const [job, setJob] = useState<JobState | null>(null)
  const [metadata, setMetadata] = useState<AnimationMetadata | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    fetchJson<Config>('/api/config')
      .then((nextConfig) => {
        setConfig(nextConfig)
        setStyle(nextConfig.defaultStyle)
        setLanguage(nextConfig.languages[0] ?? 'English')
      })
      .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)))
  }, [])

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    setAudioFile(event.target.files?.[0] ?? null)
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError('')
    setMetadata(null)

    if (mode === 'audio' && !audioFile) {
      setError('Choose an audio file before generating.')
      return
    }
    if (mode === 'text' && !text.trim()) {
      setError('Enter text before generating.')
      return
    }

    const body = new FormData()
    body.set('input_type', mode)
    body.set('style_id', style)
    body.set('clip_length', String(clipLength))
    body.set('device', device)
    body.set('text_language', language)
    if (mode === 'audio' && audioFile) body.set('audio_file', audioFile)
    if (mode === 'text') body.set('text', text)

    try {
      const created = await fetchJson<JobState>('/api/jobs', { method: 'POST', body })
      setJob(created)
      let current = created
      while (current.status === 'queued' || current.status === 'running') {
        await sleep(1200)
        current = await fetchJson<JobState>(`/api/jobs/${created.id}`)
        setJob(current)
      }
      if (current.status === 'failed') {
        throw new Error(current.error ?? 'Generation failed')
      }
      const nextMetadata = await fetchJson<AnimationMetadata>(`/api/jobs/${created.id}/metadata`)
      setMetadata(nextMetadata)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  const isBusy = job?.status === 'queued' || job?.status === 'running'

  return (
    <main className="shell">
      <section className="control-surface" aria-label="ARTalk web controls">
        <div className="brand">
          <div className="brand-mark">
            <Waves aria-hidden="true" />
          </div>
          <div>
            <p className="eyebrow">ARTalk Web Renderer</p>
            <h1>ARTalk Studio</h1>
          </div>
        </div>

        <form className="generator" onSubmit={submit}>
          <fieldset className="segmented">
            <legend>Input mode</legend>
            <button
              type="button"
              aria-pressed={mode === 'audio'}
              className={mode === 'audio' ? 'active' : ''}
              onClick={() => setMode('audio')}
            >
              <Upload aria-hidden="true" />
              Audio
            </button>
            <button
              type="button"
              aria-pressed={mode === 'text'}
              className={mode === 'text' ? 'active' : ''}
              onClick={() => setMode('text')}
            >
              <Mic2 aria-hidden="true" />
              Text
            </button>
          </fieldset>

          {mode === 'audio' ? (
            <label className="field">
              <span>Audio file</span>
              <input type="file" accept="audio/*" onChange={onFileChange} />
            </label>
          ) : (
            <label className="field">
              <span>Text prompt</span>
              <textarea value={text} onChange={(event) => setText(event.target.value)} rows={5} />
            </label>
          )}

          {mode === 'text' && (
            <label className="field">
              <span>Language</span>
              <select value={language} onChange={(event) => setLanguage(event.target.value)}>
                {config.languages.map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
            </label>
          )}

          <div className="field-grid">
            <label className="field">
              <span>Style</span>
              <select value={style} onChange={(event) => setStyle(event.target.value)}>
                {config.styles.map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Device</span>
              <select value={device} onChange={(event) => setDevice(event.target.value)}>
                <option value="auto">auto</option>
                <option value="mps">mps</option>
                <option value="cpu">cpu</option>
                <option value="cuda">cuda</option>
              </select>
            </label>
          </div>

          <label className="field">
            <span>Frame limit: {clipLength}</span>
            <input
              type="range"
              min="100"
              max="750"
              step="25"
              value={clipLength}
              onChange={(event) => setClipLength(Number(event.target.value))}
            />
          </label>

          <button type="submit" className="primary" disabled={isBusy}>
            {isBusy ? <Loader2 aria-hidden="true" /> : <Play aria-hidden="true" />}
            Generate
          </button>
        </form>

        <section className="status-panel" aria-live="polite" aria-label="Generation status">
          <span>Status</span>
          <strong>{job ? `${job.status}: ${job.stage}` : 'idle'}</strong>
          {error && <p className="error">{error}</p>}
        </section>
      </section>

      <Renderer metadata={metadata} />
    </main>
  )
}

export default App
