import { useEffect } from 'react'
import type { AnimationMetadata } from '../types'

type VideoRendererProps = {
  metadata: AnimationMetadata
  onLoadState: (state: string) => void
}

export function VideoRenderer({ metadata, onLoadState }: VideoRendererProps) {
  useEffect(() => {
    onLoadState('Loading colored video')
  }, [metadata.videoUrl, onLoadState])

  return (
    <video
      key={metadata.videoUrl}
      className="rendered-video"
      src={metadata.videoUrl ?? undefined}
      controls
      playsInline
      onCanPlay={() => onLoadState('Ready')}
      onError={() => onLoadState('Failed to load colored video')}
    />
  )
}
