import { useRef, useState } from 'react'
import { Loader2, Radio, RotateCcw } from 'lucide-react'
import type { AnimationMetadata } from '../types'
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

  return (
    <section className="stage" aria-label="Generated avatar preview">
      <div className="viewport">
        {metadata?.videoUrl ? (
          <VideoRenderer metadata={metadata} onLoadState={setLoadState} />
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
        {!metadata?.videoUrl && <audio ref={audioRef} src={metadata?.audioUrl} controls />}
        {metadata && !metadata.videoUrl && (
          <div className="renderer-tools" aria-label="Mesh renderer controls">
            <label>
              <span>Material</span>
              <select
                value={materialMode}
                onChange={(event) => setMaterialMode(event.target.value as MeshMaterialMode)}
              >
                <option value="skin">skin</option>
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
        <div className="readout" aria-live="polite">
          {metadata
            ? `${metadata.frameCount} frames · ${metadata.renderMode === 'gagavatar' ? 'colored video' : `${metadata.vertexCount} vertices`} · ${metadata.fps} fps`
            : 'No render loaded'}
        </div>
      </div>
    </section>
  )
}
