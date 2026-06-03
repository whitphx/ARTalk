export type InputMode = 'audio' | 'text'

export type AvatarInfo = {
  id: string
  label: string
  source: string
  previewUrl: string | null
}

export type Config = {
  styles: string[]
  avatars: AvatarInfo[]
  languages: string[]
  defaultStyle: string
  defaultAvatar: string
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
  fps: number
  sampleRate: number
  frameCount: number
  vertexCount: number
  faceCount: number
  verticesUrl: string
  facesUrl: string
  audioUrl: string
  motionsUrl: string
  avatarId: string
}
