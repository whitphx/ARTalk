export type InputMode = 'audio' | 'text'

export type Config = {
  styles: string[]
  languages: string[]
  defaultStyle: string
}

export type JobState = {
  id: string
  status: 'queued' | 'running' | 'complete' | 'failed'
  stage: string
  metadata?: string
  frameCount?: number
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
}
