export type InputMode = 'audio' | 'text'

export type AvatarInfo = {
  id: string
  label: string
  source: string
  previewUrl: string | null
}

export type RenderModeInfo = {
  id: 'mesh' | 'gagavatar'
  label: string
}

export type Config = {
  styles: string[]
  avatars: AvatarInfo[]
  renderModes: RenderModeInfo[]
  languages: string[]
  defaultStyle: string
  defaultAvatar: string
  defaultRenderMode: 'mesh' | 'gagavatar'
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
  renderMode: 'mesh' | 'gagavatar'
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
  avatarId: string
}
