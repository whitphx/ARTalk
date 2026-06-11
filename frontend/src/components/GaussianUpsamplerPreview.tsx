import { useEffect, useRef, useState } from 'react'
import type { AnimationMetadata } from '../types'

type GaussianUpsamplerPreviewProps = {
  metadata: AnimationMetadata
  currentTime: number
}

type RenderedUpsamplerFrames = {
  frameIndices: number[]
  images: ImageData[]
}

type OrtModule = typeof import('onnxruntime-web')
type OrtSession = import('onnxruntime-web').InferenceSession
type UpsamplerRun = {
  abortController: AbortController
  id: number
}

const UPSAMPLER_MODEL_URL = '/models/gagavatar_upsampler.onnx'
const MIN_BUFFERED_FRAMES = 3

let ortModulePromise: Promise<OrtModule> | null = null
let sessionPromise: Promise<OrtSession> | null = null
let nextRunId = 0

async function loadOrt() {
  if (!ortModulePromise) {
    ortModulePromise = import('onnxruntime-web').then((ort) => {
      ort.env.wasm.numThreads = 1
      return ort
    })
  }
  return ortModulePromise
}

async function loadUpsamplerSession() {
  const ort = await loadOrt()
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create(UPSAMPLER_MODEL_URL, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    })
  }
  return sessionPromise
}

export function GaussianUpsamplerPreview({ metadata, currentTime }: GaussianUpsamplerPreviewProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const renderedFramesRef = useRef<RenderedUpsamplerFrames | null>(null)
  const currentTimeRef = useRef(currentTime)
  const runRef = useRef<UpsamplerRun | null>(null)
  const [status, setStatus] = useState<'idle' | 'running' | 'ready' | 'failed'>('idle')
  const [message, setMessage] = useState('Upsampler')

  const inputFrameUrls = metadata.gaussianUrls?.upsamplerInputFrames
  const inputUrl = metadata.gaussianUrls?.upsamplerInputs ?? metadata.gaussianUrls?.upsamplerInputFirst
  const inputShape = metadata.gaussianUpsamplerInput?.shape
  const inputFrameCount = inputFrameUrls?.length ?? metadata.gaussianUpsamplerInput?.frameCount ?? 1
  const inputFrameIndices = metadata.gaussianUpsamplerInput?.frameIndices ?? [0]
  const canRun = Boolean(inputShape && inputFrameCount > 0 && ((inputFrameUrls?.length ?? 0) > 0 || inputUrl))

  useEffect(() => {
    currentTimeRef.current = currentTime
  }, [currentTime])

  useEffect(() => {
    runRef.current?.abortController.abort()
    runRef.current = null
    renderedFramesRef.current = null
    setStatus('idle')
    setMessage('Upsampler')
    return () => {
      runRef.current?.abortController.abort()
    }
  }, [inputFrameUrls, inputUrl])

  useEffect(() => {
    const renderedFrames = renderedFramesRef.current
    const canvas = canvasRef.current
    if (status !== 'idle' && status !== 'failed' && renderedFrames && canvas) {
      drawNearestCompletedFrame(canvas, renderedFrames, currentTime, metadata.fps)
    }
  }, [currentTime, metadata.fps, status])

  async function runUpsampler() {
    if (!inputShape || !canvasRef.current || status === 'running' || status === 'ready') return
    runRef.current?.abortController.abort()
    const run: UpsamplerRun = {
      abortController: new AbortController(),
      id: nextRunId + 1,
    }
    nextRunId = run.id
    runRef.current = run
    setStatus('running')
    setMessage('Buffering 0/0')
    try {
      const [ort, session] = await Promise.all([loadOrt(), loadUpsamplerSession()])
      const renderedFrames: RenderedUpsamplerFrames = { frameIndices: [], images: [] }
      renderedFramesRef.current = renderedFrames
      if (inputFrameUrls?.length) {
        await runFrameUrlSequence({
          urls: inputFrameUrls,
          frameIndices: inputFrameIndices,
          inputShape,
          ort,
          session,
          renderedFrames,
          canvas: canvasRef.current,
          fps: metadata.fps,
          signal: run.abortController.signal,
          getCurrentFrame: () => Math.max(0, Math.floor(currentTimeRef.current * metadata.fps)),
          onMessage: setMessage,
        })
      } else if (inputUrl) {
        await runCombinedInputSequence({
          url: inputUrl,
          frameCount: inputFrameCount,
          frameIndices: inputFrameIndices,
          inputShape,
          ort,
          session,
          renderedFrames,
          canvas: canvasRef.current,
          fps: metadata.fps,
          signal: run.abortController.signal,
          getCurrentFrame: () => Math.max(0, Math.floor(currentTimeRef.current * metadata.fps)),
          onMessage: setMessage,
        })
      }
      if (runRef.current?.id !== run.id || run.abortController.signal.aborted) return
      setStatus('ready')
      setMessage(`Ready ${renderedFrames.images.length}/${inputFrameCount}`)
    } catch (error) {
      if (run.abortController.signal.aborted) return
      setStatus('failed')
      setMessage(error instanceof Error ? error.message : String(error))
    }
  }

  return (
    <figure className="gaussian-reference gaussian-upsampler-preview">
      <canvas ref={canvasRef} width="512" height="512" aria-label="Browser ONNX upsampler first frame" />
      <figcaption>
        <button type="button" onClick={() => void runUpsampler()} disabled={!canRun || status === 'running' || status === 'ready'}>
          {status === 'running' ? 'Running' : status === 'ready' ? 'Ready' : 'Run upsampler'}
        </button>
        <span title={message}>{message}</span>
      </figcaption>
    </figure>
  )
}

async function runFrameUrlSequence({
  urls,
  frameIndices,
  inputShape,
  ort,
  session,
  renderedFrames,
  canvas,
  fps,
  signal,
  getCurrentFrame,
  onMessage,
}: {
  urls: string[]
  frameIndices: number[]
  inputShape: [number, number, number]
  ort: OrtModule
  session: OrtSession
  renderedFrames: RenderedUpsamplerFrames
  canvas: HTMLCanvasElement
  fps: number
  signal: AbortSignal
  getCurrentFrame: () => number
  onMessage: (message: string) => void
}) {
  const completed = new Set<number>()
  while (completed.size < urls.length) {
    throwIfAborted(signal)
    const index = pickNextSampleIndex(urls.length, frameIndices, completed, getCurrentFrame())
    onMessage(statusMessage('Loading', renderedFrames.images.length, urls.length))
    const inputBuffer = await fetchArrayBuffer(urls[index], signal)
    throwIfAborted(signal)
    onMessage(statusMessage('Running', renderedFrames.images.length, urls.length))
    const image = await runUpsamplerFrame(ort, session, new Uint16Array(inputBuffer), inputShape)
    throwIfAborted(signal)
    appendRenderedFrame({
      image,
      frameIndex: frameIndices[index] ?? index,
      renderedFrames,
      canvas,
      currentTime: getCurrentFrame() / fps,
      fps,
    })
    completed.add(index)
    onMessage(statusMessage('Ready', renderedFrames.images.length, urls.length))
  }
}

async function runCombinedInputSequence({
  url,
  frameCount,
  frameIndices,
  inputShape,
  ort,
  session,
  renderedFrames,
  canvas,
  fps,
  signal,
  getCurrentFrame,
  onMessage,
}: {
  url: string
  frameCount: number
  frameIndices: number[]
  inputShape: [number, number, number]
  ort: OrtModule
  session: OrtSession
  renderedFrames: RenderedUpsamplerFrames
  canvas: HTMLCanvasElement
  fps: number
  signal: AbortSignal
  getCurrentFrame: () => number
  onMessage: (message: string) => void
}) {
  onMessage('Loading')
  const values = new Uint16Array(await fetchArrayBuffer(url, signal))
  const frameValueCount = inputShape.reduce((total, value) => total * value, 1)
  if (values.length < frameValueCount * frameCount) {
    throw new Error(`Unexpected upsampler input size: ${values.length}`)
  }
  const completed = new Set<number>()
  while (completed.size < frameCount) {
    throwIfAborted(signal)
    const index = pickNextSampleIndex(frameCount, frameIndices, completed, getCurrentFrame())
    onMessage(statusMessage('Running', renderedFrames.images.length, frameCount))
    const frameValues = values.subarray(index * frameValueCount, (index + 1) * frameValueCount)
    const image = await runUpsamplerFrame(ort, session, frameValues, inputShape)
    throwIfAborted(signal)
    appendRenderedFrame({
      image,
      frameIndex: frameIndices[index] ?? index,
      renderedFrames,
      canvas,
      currentTime: getCurrentFrame() / fps,
      fps,
    })
    completed.add(index)
  }
}

async function runUpsamplerFrame(
  ort: OrtModule,
  session: OrtSession,
  frameValues: Uint16Array,
  inputShape: [number, number, number],
) {
  const input = float16ToFloat32(frameValues)
  const tensor = new ort.Tensor('float32', input, [1, ...inputShape])
  const output = await session.run({ [session.inputNames[0]]: tensor })
  const rgb = output[session.outputNames[0]]
  return rgbTensorToImageData(rgb.data as Float32Array, rgb.dims)
}

function appendRenderedFrame({
  image,
  frameIndex,
  renderedFrames,
  canvas,
  currentTime,
  fps,
}: {
  image: ImageData
  frameIndex: number
  renderedFrames: RenderedUpsamplerFrames
  canvas: HTMLCanvasElement
  currentTime: number
  fps: number
}) {
  const insertIndex = renderedFrames.frameIndices.findIndex((existingFrameIndex) => existingFrameIndex > frameIndex)
  if (insertIndex < 0) {
    renderedFrames.frameIndices.push(frameIndex)
    renderedFrames.images.push(image)
  } else {
    renderedFrames.frameIndices.splice(insertIndex, 0, frameIndex)
    renderedFrames.images.splice(insertIndex, 0, image)
  }
  drawNearestCompletedFrame(canvas, renderedFrames, currentTime, fps)
}

async function fetchArrayBuffer(url: string, signal?: AbortSignal) {
  const response = await fetch(url, { signal })
  if (!response.ok) throw new Error(`Fetch failed: ${response.status}`)
  return response.arrayBuffer()
}

function pickNextSampleIndex(
  totalCount: number,
  frameIndices: number[],
  completed: Set<number>,
  currentFrame: number,
) {
  let bestIndex = -1
  let bestDistance = Number.POSITIVE_INFINITY
  for (let index = 0; index < totalCount; index += 1) {
    if (completed.has(index)) continue
    const distance = Math.abs((frameIndices[index] ?? index) - currentFrame)
    if (distance < bestDistance) {
      bestIndex = index
      bestDistance = distance
    }
  }
  return bestIndex < 0 ? 0 : bestIndex
}

function throwIfAborted(signal: AbortSignal) {
  if (signal.aborted) {
    throw new DOMException('The operation was aborted.', 'AbortError')
  }
}

function statusMessage(phase: string, readyCount: number, totalCount: number) {
  const bufferTarget = Math.min(MIN_BUFFERED_FRAMES, totalCount)
  if (readyCount < bufferTarget) {
    return `Buffering ${readyCount}/${bufferTarget}`
  }
  return `${phase} ${readyCount}/${totalCount}`
}

function float16ToFloat32(values: Uint16Array) {
  const result = new Float32Array(values.length)
  for (let index = 0; index < values.length; index += 1) {
    result[index] = float16ToNumber(values[index])
  }
  return result
}

function float16ToNumber(value: number) {
  const sign = (value & 0x8000) ? -1 : 1
  const exponent = (value >> 10) & 0x1f
  const fraction = value & 0x03ff
  if (exponent === 0) {
    return sign * Math.pow(2, -14) * (fraction / 1024)
  }
  if (exponent === 0x1f) {
    return fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY
  }
  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024)
}

function rgbTensorToImageData(data: Float32Array, dims: readonly number[]) {
  const channels = dims[dims.length - 3]
  const height = dims[dims.length - 2]
  const width = dims[dims.length - 1]
  if (channels !== 3 || width <= 0 || height <= 0) {
    throw new Error(`Unexpected output shape: ${dims.join('x')}`)
  }
  const image = new ImageData(width, height)
  const planeSize = width * height
  for (let pixel = 0; pixel < planeSize; pixel += 1) {
    image.data[pixel * 4] = channelToByte(data[pixel])
    image.data[pixel * 4 + 1] = channelToByte(data[planeSize + pixel])
    image.data[pixel * 4 + 2] = channelToByte(data[planeSize * 2 + pixel])
    image.data[pixel * 4 + 3] = 255
  }
  return image
}

function drawNearestCompletedFrame(
  canvas: HTMLCanvasElement,
  renderedFrames: RenderedUpsamplerFrames,
  currentTime: number,
  fps: number,
) {
  if (!renderedFrames.images.length) return
  const currentFrameIndex = Math.max(0, Math.floor(currentTime * fps))
  let nearestIndex = 0
  let nearestDistance = Number.POSITIVE_INFINITY
  for (let index = 0; index < renderedFrames.frameIndices.length; index += 1) {
    const distance = Math.abs(renderedFrames.frameIndices[index] - currentFrameIndex)
    if (distance < nearestDistance) {
      nearestIndex = index
      nearestDistance = distance
    }
  }
  drawImageData(canvas, renderedFrames.images[nearestIndex])
}

function drawImageData(canvas: HTMLCanvasElement, image: ImageData) {
  canvas.width = image.width
  canvas.height = image.height
  const context = canvas.getContext('2d')
  if (!context) throw new Error('Canvas 2D context is unavailable')
  context.putImageData(image, 0, 0)
}

function channelToByte(value: number) {
  return Math.round(Math.min(Math.max(value, 0), 1) * 255)
}
