export type InputMode = 'audio' | 'text'

export type AvatarInfo = {
  id: string
  label: string
  source: string
  previewUrl: string | null
}

export type RenderModeInfo = {
  id: 'mesh' | 'browser-gaussian' | 'gagavatar'
  label: string
}

export type Config = {
  styles: string[]
  avatars: AvatarInfo[]
  renderModes: RenderModeInfo[]
  languages: string[]
  defaultStyle: string
  defaultAvatar: string
  defaultRenderMode: 'mesh' | 'browser-gaussian' | 'gagavatar'
}

export type JobState = {
  id: string
  status: 'queued' | 'running' | 'complete' | 'failed'
  stage: string
  metadata?: string
  frameCount?: number
  avatarId?: string
  error?: string
}

export type AnimationMetadata = {
  renderMode: 'mesh' | 'browser-gaussian' | 'gagavatar'
  fps: number
  sampleRate: number
  frameCount: number
  vertexCount: number
  faceCount: number
  verticesUrl: string
  facesUrl: string
  regionLabelsUrl?: string | null
  regionLabelFormat?: 'uint8-vertex'
  regionLabels?: Record<string, number>
  regionSource?: string
  audioUrl: string
  motionsUrl: string
  videoUrl: string | null
  gaussianCount?: number
  gaussianFormat?: 'gagavatar-first-frame-f32-v1'
  gaussianColorChannels?: number
  gaussianUrls?: {
    xyz: string
    colors: string
    opacities: string
    scales: string
    rotations: string
  }
  gaussianCamera?: {
    focalX: number
    focalY: number
    size: [number, number]
  }
  avatarId: string
}
