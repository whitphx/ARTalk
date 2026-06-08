import { useEffect, useRef, useState } from 'react'
import { Loader2, Pause, Play, Radio, RotateCcw } from 'lucide-react'
import type { AnimationMetadata } from '../types'
import { GaussianPointRenderer, type GaussianPreviewMode } from './GaussianPointRenderer'
import { MeshFaceRenderer, type MeshMaterialMode } from './MeshFaceRenderer'
import { VideoRenderer } from './VideoRenderer'

type RendererProps = {
  metadata: AnimationMetadata | null
}

export function Renderer({ metadata }: RendererProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [loadState, setLoadState] = useState('Waiting for animation data')
  const [materialMode, setMaterialMode] = useState<MeshMaterialMode>('skin')
  const [wireframe, setWireframe] = useState(false)
  const [cameraResetSignal, setCameraResetSignal] = useState(0)
  const [gaussianPreviewMode, setGaussianPreviewMode] = useState<GaussianPreviewMode>('head')
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const isMeshRender = metadata?.renderMode === 'mesh'
  const isGaussianRender = metadata?.renderMode === 'browser-gaussian'
  const duration = metadata ? metadata.frameCount / metadata.fps : 0
  const currentFrame = metadata
    ? Math.min(metadata.frameCount, Math.floor(currentTime * metadata.fps) + 1)
    : 0

  useEffect(() => {
    const audio = audioRef.current
    audio?.pause()
    if (audio) audio.currentTime = 0
    setCurrentTime(0)
    setIsPlaying(false)
  }, [metadata?.audioUrl])

  async function togglePlayback() {
    const audio = audioRef.current
    if (!audio || !metadata) return
    if (!audio.paused) {
      audio.pause()
      setIsPlaying(false)
      return
    }
    if (audio.currentTime >= duration) audio.currentTime = 0
    await audio.play()
    setIsPlaying(true)
  }

  function seekTo(value: number) {
    const nextTime = Math.min(Math.max(value, 0), duration)
    const audio = audioRef.current
    if (audio) audio.currentTime = nextTime
    setCurrentTime(nextTime)
  }

  return (
    <section className="stage" aria-label="Generated avatar preview">
      <div className="viewport">
        {metadata?.videoUrl ? (
          <VideoRenderer metadata={metadata} onLoadState={setLoadState} />
        ) : isGaussianRender ? (
          <GaussianPointRenderer
            metadata={metadata}
            audioRef={audioRef}
            previewMode={gaussianPreviewMode}
            onLoadState={setLoadState}
          />
        ) : (
          <MeshFaceRenderer
            metadata={metadata}
            audioRef={audioRef}
            materialMode={materialMode}
            wireframe={wireframe}
            cameraResetSignal={cameraResetSignal}
            onLoadState={setLoadState}
          />
        )}
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
        {(isMeshRender || isGaussianRender) && (
          <div className="playback-tools" aria-label="Playback controls">
            <audio
              ref={audioRef}
              src={metadata?.audioUrl}
              preload="auto"
              onPause={() => setIsPlaying(false)}
              onPlay={() => setIsPlaying(true)}
              onEnded={() => {
                setIsPlaying(false)
                setCurrentTime(duration)
              }}
              onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
            />
            <button
              type="button"
              className="icon-command"
              aria-label={isPlaying ? 'Pause' : 'Play'}
              title={isPlaying ? 'Pause' : 'Play'}
              onClick={() => void togglePlayback()}
              disabled={loadState !== 'Ready'}
            >
              {isPlaying ? <Pause aria-hidden="true" /> : <Play aria-hidden="true" />}
              {isPlaying ? 'Pause' : 'Play'}
            </button>
            <label className="scrubber">
              <span>Timeline</span>
              <input
                type="range"
                min="0"
                max={duration || 0}
                step={metadata ? 1 / metadata.fps : 0.04}
                value={Math.min(currentTime, duration)}
                onChange={(event) => seekTo(Number(event.target.value))}
                disabled={loadState !== 'Ready'}
              />
            </label>
            <span className="timecode">{formatTime(currentTime)} / {formatTime(duration)}</span>
          </div>
        )}
        {isMeshRender && (
          <div className="renderer-tools" aria-label="Mesh renderer controls">
            <label>
              <span>Material</span>
              <select
                value={materialMode}
                onChange={(event) => setMaterialMode(event.target.value as MeshMaterialMode)}
              >
                <option value="skin">skin</option>
                <option value="region">region</option>
                <option value="debug">debug</option>
                <option value="normal">normal</option>
              </select>
            </label>
            <label className="toggle-field">
              <input
                type="checkbox"
                checked={wireframe}
                onChange={(event) => setWireframe(event.target.checked)}
              />
              <span>Wireframe</span>
            </label>
            <button
              type="button"
              className="icon-command"
              aria-label="Reset camera"
              title="Reset camera"
              onClick={() => setCameraResetSignal((value) => value + 1)}
            >
              <RotateCcw aria-hidden="true" />
              Reset camera
            </button>
          </div>
        )}
        {isGaussianRender && (
          <div className="renderer-tools" aria-label="Gaussian renderer controls">
            <label>
              <span>Preview</span>
              <select
                value={gaussianPreviewMode}
                onChange={(event) => setGaussianPreviewMode(event.target.value as GaussianPreviewMode)}
              >
                <option value="head">head</option>
                <option value="planes">planes</option>
                <option value="all">all</option>
              </select>
            </label>
          </div>
        )}
        <div className="readout" aria-live="polite">
          {metadata
            ? `${metadata.renderMode === 'gagavatar' ? 'colored video' : metadata.renderMode === 'browser-gaussian' ? `${metadata.gaussianCount ?? 0} gaussians` : `frame ${currentFrame}/${metadata.frameCount} · ${metadata.vertexCount} vertices`} · ${metadata.fps} fps`
            : 'No render loaded'}
        </div>
      </div>
    </section>
  )
}

function formatTime(value: number) {
  const safeValue = Number.isFinite(value) ? Math.max(value, 0) : 0
  const minutes = Math.floor(safeValue / 60)
  const seconds = Math.floor(safeValue % 60)
  const centiseconds = Math.floor((safeValue % 1) * 100)
  return `${minutes}:${seconds.toString().padStart(2, '0')}.${centiseconds.toString().padStart(2, '0')}`
}
