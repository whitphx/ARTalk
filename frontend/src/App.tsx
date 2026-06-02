import { useEffect, useState } from 'react'
import type { ChangeEvent, FormEvent } from 'react'
import { Loader2, Mic2, Play, Upload, Waves } from 'lucide-react'
import { fetchJson, sleep } from './api'
import { Renderer } from './components/Renderer'
import type { AnimationMetadata, Config, InputMode, JobState } from './types'
import './App.css'

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
