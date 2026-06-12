import { useEffect, useState } from 'react'
import type { ChangeEvent, FormEvent } from 'react'
import { Check, Copy, ImageUp, Loader2, Mic2, Play, Upload, Waves } from 'lucide-react'
import { fetchJson, sleep } from './api'
import { Renderer } from './components/Renderer'
import type { AnimationMetadata, Config, InputMode, JobState, RenderModeInfo } from './types'
import './App.css'

const DEFAULT_AVATARS = [{ id: 'mesh', label: 'Neutral mesh', source: 'mesh', previewUrl: null }]
const DEFAULT_RENDER_MODES = [
  { id: 'mesh' as const, label: 'Browser mesh' },
  { id: 'browser-gaussian' as const, label: 'Browser Gaussian (experimental)' },
  { id: 'gagavatar' as const, label: 'Colored video (server)' },
]
type RenderModeId = RenderModeInfo['id']

function normalizeConfig(nextConfig: Config): Config {
  return {
    styles: nextConfig.styles ?? ['default'],
    avatars: nextConfig.avatars ?? DEFAULT_AVATARS,
    renderModes: nextConfig.renderModes ?? DEFAULT_RENDER_MODES,
    languages: nextConfig.languages ?? ['English'],
    defaultStyle: nextConfig.defaultStyle ?? 'default',
    defaultAvatar: nextConfig.defaultAvatar ?? 'mesh',
    defaultRenderMode: nextConfig.defaultRenderMode ?? 'mesh',
  }
}

function App() {
  const [config, setConfig] = useState<Config>({
    styles: ['default'],
    avatars: DEFAULT_AVATARS,
    renderModes: DEFAULT_RENDER_MODES,
    languages: ['English'],
    defaultStyle: 'default',
    defaultAvatar: 'mesh',
    defaultRenderMode: 'mesh',
  })
  const [mode, setMode] = useState<InputMode>('audio')
  const [audioFile, setAudioFile] = useState<File | null>(null)
  const [avatarImage, setAvatarImage] = useState<File | null>(null)
  const [text, setText] = useState('')
  const [language, setLanguage] = useState('English')
  const [style, setStyle] = useState('default')
  const [avatar, setAvatar] = useState('mesh')
  const [renderMode, setRenderMode] = useState<RenderModeId>('mesh')
  const [clipLength, setClipLength] = useState(300)
  const [device, setDevice] = useState('auto')
  const [job, setJob] = useState<JobState | null>(null)
  const [avatarJob, setAvatarJob] = useState<JobState | null>(null)
  const [metadata, setMetadata] = useState<AnimationMetadata | null>(null)
  const [error, setError] = useState('')
  const [errorCopyState, setErrorCopyState] = useState<'idle' | 'copied' | 'failed'>('idle')

  function showError(message: string) {
    setError(message)
    setErrorCopyState('idle')
  }

  function clearError() {
    setError('')
    setErrorCopyState('idle')
  }

  useEffect(() => {
    fetchJson<Config>('/api/config')
      .then((nextConfig) => {
        const normalized = normalizeConfig(nextConfig)
        setConfig(normalized)
        setStyle(normalized.defaultStyle)
        setAvatar(normalized.defaultAvatar)
        setRenderMode(normalized.defaultRenderMode)
        setLanguage(normalized.languages[0] ?? 'English')
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : String(err))
        setErrorCopyState('idle')
      })
  }, [])

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    setAudioFile(event.target.files?.[0] ?? null)
  }

  async function copyError() {
    try {
      await navigator.clipboard.writeText(error)
      setErrorCopyState('copied')
      window.setTimeout(() => setErrorCopyState('idle'), 1800)
    } catch {
      setErrorCopyState('failed')
      window.setTimeout(() => setErrorCopyState('idle'), 2200)
    }
  }

  function onAvatarImageChange(event: ChangeEvent<HTMLInputElement>) {
    setAvatarImage(event.target.files?.[0] ?? null)
  }

  async function refreshAvatars(nextAvatarId?: string) {
    const nextConfig = normalizeConfig(await fetchJson<Config>('/api/config'))
    setConfig(nextConfig)
    setAvatar(nextAvatarId ?? nextConfig.defaultAvatar)
  }

  async function registerAvatar() {
    clearError()
    if (!avatarImage) {
      showError('Choose a face image before registering an avatar.')
      return
    }

    const body = new FormData()
    body.set('device', device)
    body.set('image_file', avatarImage)

    try {
      const created = await fetchJson<JobState>('/api/avatar-jobs', { method: 'POST', body })
      setAvatarJob(created)
      let current = created
      while (current.status === 'queued' || current.status === 'running') {
        await sleep(1200)
        current = await fetchJson<JobState>(`/api/avatar-jobs/${created.id}`)
        setAvatarJob(current)
      }
      if (current.status === 'failed') {
        throw new Error(current.error ?? 'Avatar registration failed')
      }
      await refreshAvatars(current.avatarId)
    } catch (err: unknown) {
      showError(err instanceof Error ? err.message : String(err))
    }
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    clearError()
    setMetadata(null)

    if (mode === 'audio' && !audioFile) {
      showError('Choose an audio file before generating.')
      return
    }
    if (mode === 'text' && !text.trim()) {
      showError('Enter text before generating.')
      return
    }

    const body = new FormData()
    body.set('input_type', mode)
    body.set('style_id', style)
    body.set('clip_length', String(clipLength))
    body.set('device', device)
    body.set('avatar_id', avatar)
    body.set('render_mode', renderMode)
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
      showError(err instanceof Error ? err.message : String(err))
    }
  }

  const isBusy = job?.status === 'queued' || job?.status === 'running'
  const isAvatarBusy = avatarJob?.status === 'queued' || avatarJob?.status === 'running'

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
            <span>Avatar</span>
            <select value={avatar} onChange={(event) => setAvatar(event.target.value)}>
              {config.avatars.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.label}
                </option>
              ))}
            </select>
          </label>

          <label className="field">
            <span>Output</span>
            <select
              value={renderMode}
              onChange={(event) => setRenderMode(event.target.value as RenderModeId)}
            >
              {config.renderModes.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.label}
                </option>
              ))}
            </select>
          </label>

          <div className="avatar-uploader">
            <label className="field">
              <span>Face image</span>
              <input type="file" accept="image/png,image/jpeg" onChange={onAvatarImageChange} />
            </label>
            <button type="button" onClick={registerAvatar} disabled={isAvatarBusy}>
              {isAvatarBusy ? <Loader2 aria-hidden="true" /> : <ImageUp aria-hidden="true" />}
              Register
            </button>
          </div>

          <label className="field">
            <span>Frame limit: {clipLength}</span>
            <input
              type="range"
              min="50"
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
          {avatarJob && <small>Avatar: {avatarJob.status}: {avatarJob.stage}</small>}
          {error && (
            <div className="error-panel">
              <div className="error-toolbar">
                <span>Error</span>
                <button type="button" className="copy-error" onClick={copyError}>
                  {errorCopyState === 'copied' ? (
                    <Check aria-hidden="true" />
                  ) : (
                    <Copy aria-hidden="true" />
                  )}
                  {errorCopyState === 'failed' ? 'Copy failed' : errorCopyState === 'copied' ? 'Copied' : 'Copy'}
                </button>
              </div>
              <p className="error">{error}</p>
            </div>
          )}
        </section>
      </section>

      <Renderer metadata={metadata} />
    </main>
  )
}

export default App
