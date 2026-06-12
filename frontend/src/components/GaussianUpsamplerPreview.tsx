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
type UpsamplerRuntimeInfo = {
  crossOriginIsolated: boolean
  numThreads: number
  provider: 'webgpu' | 'wasm'
  fallbackReason?: string
}
type UpsamplerFrameResult = {
  image: ImageData
  inputMs: number
  inferenceMs: number
  outputMs: number
}
type UpsamplerStats = {
  startedAt: number
  fetchMs: number
  inputMs: number
  inferenceMs: number
  outputMs: number
  runtime: UpsamplerRuntimeInfo
}
type UpsamplerRun = {
  abortController: AbortController
  id: number
}

const UPSAMPLER_MODEL_URL = '/models/gagavatar_upsampler.onnx'
const MIN_BUFFERED_FRAMES = 3
const MAX_WASM_THREADS = 4

let ortModulePromise: Promise<OrtModule> | null = null
let sessionPromise: Promise<OrtSession> | null = null
let runtimeInfo: UpsamplerRuntimeInfo | null = null
let nextRunId = 0

async function loadOrt() {
  if (!ortModulePromise) {
    ortModulePromise = loadPreferredOrtModule().then((ort) => {
      runtimeInfo = configureOrtRuntime(ort)
      return ort
    })
  }
  return ortModulePromise
}

async function loadUpsamplerSession() {
  const ort = await loadOrt()
  if (!sessionPromise) {
    sessionPromise = createUpsamplerSession(ort)
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
      const stats = createUpsamplerStats()
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
          stats,
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
          stats,
        })
      }
      if (runRef.current?.id !== run.id || run.abortController.signal.aborted) return
      setStatus('ready')
      setMessage(statusMessage('Ready', renderedFrames.images.length, inputFrameCount, stats))
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
  stats,
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
  stats: UpsamplerStats
}) {
  const completed = new Set<number>()
  while (completed.size < urls.length) {
    throwIfAborted(signal)
    const index = pickNextSampleIndex(urls.length, frameIndices, completed, getCurrentFrame())
    onMessage(statusMessage('Loading', renderedFrames.images.length, urls.length, stats))
    const fetchStart = performance.now()
    const inputBuffer = await fetchArrayBuffer(urls[index], signal)
    stats.fetchMs += performance.now() - fetchStart
    throwIfAborted(signal)
    onMessage(statusMessage('Running', renderedFrames.images.length, urls.length, stats))
    const result = await runUpsamplerFrame(ort, session, new Uint16Array(inputBuffer), inputShape)
    stats.inputMs += result.inputMs
    stats.inferenceMs += result.inferenceMs
    stats.outputMs += result.outputMs
    throwIfAborted(signal)
    appendRenderedFrame({
      image: result.image,
      frameIndex: frameIndices[index] ?? index,
      renderedFrames,
      canvas,
      currentTime: getCurrentFrame() / fps,
      fps,
    })
    completed.add(index)
    onMessage(statusMessage('Ready', renderedFrames.images.length, urls.length, stats))
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
  stats,
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
  stats: UpsamplerStats
}) {
  onMessage('Loading')
  const fetchStart = performance.now()
  const values = new Uint16Array(await fetchArrayBuffer(url, signal))
  stats.fetchMs += performance.now() - fetchStart
  const frameValueCount = inputShape.reduce((total, value) => total * value, 1)
  if (values.length < frameValueCount * frameCount) {
    throw new Error(`Unexpected upsampler input size: ${values.length}`)
  }
  const completed = new Set<number>()
  while (completed.size < frameCount) {
    throwIfAborted(signal)
    const index = pickNextSampleIndex(frameCount, frameIndices, completed, getCurrentFrame())
    onMessage(statusMessage('Running', renderedFrames.images.length, frameCount, stats))
    const frameValues = values.subarray(index * frameValueCount, (index + 1) * frameValueCount)
    const result = await runUpsamplerFrame(ort, session, frameValues, inputShape)
    stats.inputMs += result.inputMs
    stats.inferenceMs += result.inferenceMs
    stats.outputMs += result.outputMs
    throwIfAborted(signal)
    appendRenderedFrame({
      image: result.image,
      frameIndex: frameIndices[index] ?? index,
      renderedFrames,
      canvas,
      currentTime: getCurrentFrame() / fps,
      fps,
    })
    completed.add(index)
    onMessage(statusMessage('Ready', renderedFrames.images.length, frameCount, stats))
  }
}

async function runUpsamplerFrame(
  ort: OrtModule,
  session: OrtSession,
  frameValues: Uint16Array,
  inputShape: [number, number, number],
): Promise<UpsamplerFrameResult> {
  const inputStart = performance.now()
  const input = float16ToFloat32(frameValues)
  const tensor = new ort.Tensor('float32', input, [1, ...inputShape])
  const inputMs = performance.now() - inputStart
  const inferenceStart = performance.now()
  const output = await session.run({ [session.inputNames[0]]: tensor })
  const inferenceMs = performance.now() - inferenceStart
  const outputStart = performance.now()
  const rgb = output[session.outputNames[0]]
  const image = rgbTensorToImageData(rgb.data as Float32Array, rgb.dims)
  return {
    image,
    inputMs,
    inferenceMs,
    outputMs: performance.now() - outputStart,
  }
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

function configureOrtRuntime(ort: OrtModule): UpsamplerRuntimeInfo {
  const isolated = Boolean(globalThis.crossOriginIsolated)
  const hardwareConcurrency = navigator.hardwareConcurrency || 1
  const numThreads = isolated ? Math.min(MAX_WASM_THREADS, Math.max(2, Math.ceil(hardwareConcurrency / 2))) : 1
  ort.env.wasm.numThreads = numThreads
  return {
    crossOriginIsolated: isolated,
    numThreads,
    provider: hasWebGpuSupport() ? 'webgpu' : 'wasm',
    fallbackReason: hasWebGpuSupport() ? undefined : 'WebGPU unavailable',
  }
}

async function loadPreferredOrtModule(): Promise<OrtModule> {
  if (hasWebGpuSupport()) return import('onnxruntime-web/webgpu')
  return import('onnxruntime-web')
}

async function createUpsamplerSession(ort: OrtModule): Promise<OrtSession> {
  if (hasWebGpuSupport()) {
    try {
      const session = await ort.InferenceSession.create(UPSAMPLER_MODEL_URL, {
        executionProviders: [{ name: 'webgpu', preferredLayout: 'NCHW' }],
        graphOptimizationLevel: 'all',
      })
      runtimeInfo = {
        ...(runtimeInfo ?? configureOrtRuntime(ort)),
        provider: 'webgpu',
        fallbackReason: undefined,
      }
      return session
    } catch (error) {
      runtimeInfo = {
        ...(runtimeInfo ?? configureOrtRuntime(ort)),
        provider: 'wasm',
        fallbackReason: error instanceof Error ? error.message : String(error),
      }
      console.warn('Falling back to WASM upsampler runtime after WebGPU session creation failed.', error)
    }
  }
  const session = await ort.InferenceSession.create(UPSAMPLER_MODEL_URL, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  })
  runtimeInfo = {
    ...(runtimeInfo ?? configureOrtRuntime(ort)),
    provider: 'wasm',
  }
  return session
}

function hasWebGpuSupport() {
  return Boolean(globalThis.isSecureContext && navigator.gpu)
}

function createUpsamplerStats(): UpsamplerStats {
  return {
    startedAt: performance.now(),
    fetchMs: 0,
    inputMs: 0,
    inferenceMs: 0,
    outputMs: 0,
    runtime: runtimeInfo ?? {
      crossOriginIsolated: Boolean(globalThis.crossOriginIsolated),
      numThreads: 1,
      provider: hasWebGpuSupport() ? 'webgpu' : 'wasm',
      fallbackReason: hasWebGpuSupport() ? undefined : 'WebGPU unavailable',
    },
  }
}

function statusMessage(phase: string, readyCount: number, totalCount: number, stats: UpsamplerStats) {
  const bufferTarget = Math.min(MIN_BUFFERED_FRAMES, totalCount)
  const runtime = runtimeLabel(stats.runtime)
  if (readyCount > 0) {
    const elapsedSeconds = Math.max((performance.now() - stats.startedAt) / 1000, 0.001)
    const fps = readyCount / elapsedSeconds
    const measuredMs = (stats.fetchMs + stats.inputMs + stats.inferenceMs + stats.outputMs) / readyCount
    const inferenceMs = stats.inferenceMs / readyCount
    return `${phase} ${readyCount}/${totalCount} · ${fps.toFixed(1)} fps · ${measuredMs.toFixed(0)} ms/f · infer ${inferenceMs.toFixed(0)} ms · ${runtime}`
  }
  if (readyCount < bufferTarget) {
    return `Buffering ${readyCount}/${bufferTarget} · ${runtime}`
  }
  return `${phase} ${readyCount}/${totalCount} · ${runtime}`
}

function runtimeLabel(runtime: UpsamplerRuntimeInfo) {
  if (runtime.provider === 'webgpu') return 'webgpu'
  const fallback = runtime.fallbackReason ? ' fallback' : ''
  return `wasm ${runtime.numThreads}t${runtime.crossOriginIsolated ? '' : ' no-isolation'}${fallback}`
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
