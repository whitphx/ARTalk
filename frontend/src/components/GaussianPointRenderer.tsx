import { useEffect, useRef } from 'react'
import type { RefObject } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { AnimationMetadata } from '../types'

type GaussianPointRendererProps = {
  metadata: AnimationMetadata
  audioRef: RefObject<HTMLAudioElement | null>
  previewMode: GaussianPreviewMode
  onLoadState: (state: string) => void
}

export type GaussianPreviewMode = 'head' | 'planes' | 'all'

const GAGAVATAR_HEAD_GAUSSIAN_COUNT = 5023

const SPLAT_VERTEX_SHADER = `
attribute vec3 center;
attribute vec3 gaussianColor;
attribute float gaussianOpacity;
attribute vec3 gaussianScale;
attribute vec4 gaussianRotation;
varying vec3 vColor;
varying float vOpacity;
varying vec2 vQuad;

vec3 rotateByQuaternion(vec3 value, vec4 quaternion) {
  return value + 2.0 * cross(quaternion.xyz, cross(quaternion.xyz, value) + quaternion.w * value);
}

void main() {
  vColor = gaussianColor;
  vOpacity = gaussianOpacity;
  vQuad = position.xy;
  vec3 axisX = rotateByQuaternion(vec3(gaussianScale.x, 0.0, 0.0), gaussianRotation);
  vec3 axisY = rotateByQuaternion(vec3(0.0, gaussianScale.y, 0.0), gaussianRotation);
  vec3 splatPosition = center + axisX * position.x * 2.6 + axisY * position.y * 2.6;
  vec4 viewPosition = modelViewMatrix * vec4(splatPosition, 1.0);
  gl_Position = projectionMatrix * viewPosition;
}
`

const SPLAT_FRAGMENT_SHADER = `
varying vec3 vColor;
varying float vOpacity;
varying vec2 vQuad;

void main() {
  float radius2 = dot(vQuad, vQuad);
  if (radius2 > 1.0) discard;
  float alpha = exp(-radius2 * 3.2) * vOpacity;
  gl_FragColor = vec4(vColor, alpha);
}
`

export function GaussianPointRenderer({
  metadata,
  audioRef,
  previewMode,
  onLoadState,
}: GaussianPointRendererProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    if (!canvasRef.current) return
    if (!metadata.gaussianUrls || !metadata.gaussianCount || !metadata.gaussianColorChannels) {
      onLoadState('Gaussian snapshot is not available')
      return
    }

    let disposed = false
    let animationId = 0
    let resizeFrame = 0
    let resizeObserver: ResizeObserver | null = null
    let activeHeadFrame = -1
    const canvas = canvasRef.current
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true })
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.0

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(28, 1, 0.01, 100)
    const controls = new OrbitControls(camera, canvas)
    controls.enableDamping = true
    controls.enablePan = false
    controls.minDistance = 0.35
    controls.maxDistance = 7.5

    async function load() {
      onLoadState('Loading Gaussian buffers')
      const shouldLoadAnimatedHead = previewMode === 'head' && Boolean(metadata.gaussianUrls!.headXyz)
      const [xyzResponse, headXyzResponse, colorResponse, opacityResponse, scaleResponse, rotationResponse] = await Promise.all([
        fetch(metadata.gaussianUrls!.xyz),
        shouldLoadAnimatedHead ? fetch(metadata.gaussianUrls!.headXyz!) : Promise.resolve(null),
        fetch(metadata.gaussianUrls!.colors),
        fetch(metadata.gaussianUrls!.opacities),
        fetch(metadata.gaussianUrls!.scales),
        fetch(metadata.gaussianUrls!.rotations),
      ])
      if (!xyzResponse.ok) throw new Error(`Failed to load Gaussian positions: ${xyzResponse.status}`)
      if (headXyzResponse && !headXyzResponse.ok) {
        throw new Error(`Failed to load animated Gaussian head positions: ${headXyzResponse.status}`)
      }
      if (!colorResponse.ok) throw new Error(`Failed to load Gaussian colors: ${colorResponse.status}`)
      if (!opacityResponse.ok) throw new Error(`Failed to load Gaussian opacities: ${opacityResponse.status}`)
      if (!scaleResponse.ok) throw new Error(`Failed to load Gaussian scales: ${scaleResponse.status}`)
      if (!rotationResponse.ok) throw new Error(`Failed to load Gaussian rotations: ${rotationResponse.status}`)

      const [xyzBuffer, headXyzBuffer, colorBuffer, opacityBuffer, scaleBuffer, rotationBuffer] = await Promise.all([
        xyzResponse.arrayBuffer(),
        headXyzResponse ? headXyzResponse.arrayBuffer() : Promise.resolve(null),
        colorResponse.arrayBuffer(),
        opacityResponse.arrayBuffer(),
        scaleResponse.arrayBuffer(),
        rotationResponse.arrayBuffer(),
      ])
      if (disposed) return

      const sourcePositions = new Float32Array(xyzBuffer)
      const animatedHeadPositions = headXyzBuffer ? new Float32Array(headXyzBuffer) : null
      const sourceColors = new Float32Array(colorBuffer)
      const sourceOpacities = new Float32Array(opacityBuffer)
      const sourceScales = new Float32Array(scaleBuffer)
      const sourceRotations = new Float32Array(rotationBuffer)
      const count = metadata.gaussianCount!
      const colorChannels = metadata.gaussianColorChannels!
      const headCount = Math.min(metadata.gaussianHeadCount ?? GAGAVATAR_HEAD_GAUSSIAN_COUNT, count)
      if (sourcePositions.length !== count * 3) throw new Error('Invalid Gaussian position buffer size')
      if (
        animatedHeadPositions &&
        animatedHeadPositions.length !== (metadata.gaussianHeadFrameCount ?? metadata.frameCount) * headCount * 3
      ) {
        throw new Error('Invalid animated Gaussian head position buffer size')
      }
      if (sourceColors.length !== count * colorChannels) throw new Error('Invalid Gaussian color buffer size')
      if (sourceOpacities.length !== count) throw new Error('Invalid Gaussian opacity buffer size')
      if (sourceScales.length !== count * 3) throw new Error('Invalid Gaussian scale buffer size')
      if (sourceRotations.length !== count * 4) throw new Error('Invalid Gaussian rotation buffer size')

      const range = gaussianPreviewRange(previewMode, count, headCount)
      const previewCount = range.end - range.start
      const centers = new Float32Array(previewCount * 3)
      const previewColors = new Float32Array(previewCount * 3)
      const previewOpacities = new Float32Array(previewCount)
      const previewScales = new Float32Array(previewCount * 3)
      const previewRotations = new Float32Array(previewCount * 4)
      for (let index = 0; index < previewCount; index += 1) {
        const sourceIndex = range.start + index
        const sourceOffset = sourceIndex * 3
        centers[index * 3] = sourcePositions[sourceOffset]
        centers[index * 3 + 1] = sourcePositions[sourceOffset + 1]
        centers[index * 3 + 2] = sourcePositions[sourceOffset + 2]
        const colorOffset = sourceIndex * colorChannels
        const opacity = Math.min(Math.max(sourceOpacities[sourceIndex], 0.02), 1)
        const scaleOffset = sourceIndex * 3
        const previewOffset = index * 3
        previewColors[previewOffset] = sigmoid(sourceColors[colorOffset])
        previewColors[previewOffset + 1] = sigmoid(sourceColors[colorOffset + 1])
        previewColors[previewOffset + 2] = sigmoid(sourceColors[colorOffset + 2])
        previewOpacities[index] = Math.max(opacity, 0.12)
        previewScales[previewOffset] = Math.max(sourceScales[scaleOffset], 0.006)
        previewScales[previewOffset + 1] = Math.max(sourceScales[scaleOffset + 1], 0.006)
        previewScales[previewOffset + 2] = Math.max(sourceScales[scaleOffset + 2], 0.006)
        const rotationOffset = sourceIndex * 4
        const previewRotationOffset = index * 4
        previewRotations[previewRotationOffset] = sourceRotations[rotationOffset]
        previewRotations[previewRotationOffset + 1] = sourceRotations[rotationOffset + 1]
        previewRotations[previewRotationOffset + 2] = sourceRotations[rotationOffset + 2]
        previewRotations[previewRotationOffset + 3] = sourceRotations[rotationOffset + 3]
      }

      const geometry = new THREE.InstancedBufferGeometry()
      geometry.instanceCount = previewCount
      geometry.setAttribute('position', new THREE.BufferAttribute(buildUnitQuadVertices(), 3))
      geometry.setAttribute('center', new THREE.InstancedBufferAttribute(centers, 3))
      geometry.setAttribute('gaussianColor', new THREE.InstancedBufferAttribute(previewColors, 3))
      geometry.setAttribute('gaussianOpacity', new THREE.InstancedBufferAttribute(previewOpacities, 1))
      geometry.setAttribute('gaussianScale', new THREE.InstancedBufferAttribute(previewScales, 3))
      geometry.setAttribute('gaussianRotation', new THREE.InstancedBufferAttribute(previewRotations, 4))
      geometry.boundingSphere = boundingSphereForCenters(centers)

      const material = new THREE.ShaderMaterial({
        vertexShader: SPLAT_VERTEX_SHADER,
        fragmentShader: SPLAT_FRAGMENT_SHADER,
        transparent: true,
        blending: THREE.NormalBlending,
        depthWrite: false,
        depthTest: false,
      })
      const points = new THREE.Mesh(geometry, material)
      scene.add(points)

      const sphere = geometry.boundingSphere
      if (sphere) {
        points.position.set(-sphere.center.x, -sphere.center.y, -sphere.center.z)
        camera.position.set(0, 0.02, Math.max(sphere.radius * 3.8, 1.2))
        controls.target.set(0, 0, 0)
        controls.update()
      }

      function resize() {
        const parent = canvas.parentElement
        if (!parent) return
        const { width, height } = parent.getBoundingClientRect()
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
        renderer.setSize(width, height, false)
        camera.aspect = width / Math.max(height, 1)
        camera.updateProjectionMatrix()
      }

      function scheduleResize() {
        window.cancelAnimationFrame(resizeFrame)
        resizeFrame = window.requestAnimationFrame(resize)
      }

      function renderLoop() {
        if (disposed) return
        updateAnimatedHeadCenters(
          centers,
          geometry.attributes.center,
          animatedHeadPositions,
          audioRef.current?.currentTime ?? 0,
          metadata.fps,
          headCount,
          activeHeadFrame,
          (frame) => {
            activeHeadFrame = frame
          },
        )
        controls.update()
        renderer.render(scene, camera)
        animationId = window.requestAnimationFrame(renderLoop)
      }

      resizeObserver = new ResizeObserver(scheduleResize)
      resizeObserver.observe(canvas.parentElement ?? canvas)
      resize()
      onLoadState('Ready')
      renderLoop()
    }

    load().catch((error: unknown) => {
      if (!disposed) onLoadState(error instanceof Error ? error.message : String(error))
    })

    return () => {
      disposed = true
      window.cancelAnimationFrame(animationId)
      window.cancelAnimationFrame(resizeFrame)
      resizeObserver?.disconnect()
      controls.dispose()
      renderer.dispose()
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh) {
          object.geometry.dispose()
          const materials = Array.isArray(object.material) ? object.material : [object.material]
          materials.forEach((material) => material.dispose())
        }
      })
    }
  }, [metadata, audioRef, previewMode, onLoadState])

  return <canvas ref={canvasRef} aria-label="Experimental Gaussian avatar renderer" />
}

function sigmoid(value: number) {
  return 1 / (1 + Math.exp(-value))
}

function gaussianPreviewRange(previewMode: GaussianPreviewMode, count: number, headCount: number) {
  if (previewMode === 'head') return { start: 0, end: headCount }
  if (previewMode === 'planes') return { start: headCount, end: count }
  return { start: 0, end: count }
}

function buildUnitQuadVertices() {
  return new Float32Array([
    -1, -1, 0,
    1, -1, 0,
    -1, 1, 0,
    -1, 1, 0,
    1, -1, 0,
    1, 1, 0,
  ])
}

function boundingSphereForCenters(centers: Float32Array) {
  const box = new THREE.Box3()
  const point = new THREE.Vector3()
  for (let index = 0; index < centers.length; index += 3) {
    point.set(centers[index], centers[index + 1], centers[index + 2])
    box.expandByPoint(point)
  }
  const sphere = new THREE.Sphere()
  box.getBoundingSphere(sphere)
  sphere.radius += 0.2
  return sphere
}

function updateAnimatedHeadCenters(
  centers: Float32Array,
  centerAttribute: THREE.BufferAttribute | THREE.InterleavedBufferAttribute,
  animatedHeadPositions: Float32Array | null,
  currentTime: number,
  fps: number,
  headCount: number,
  activeFrame: number,
  setActiveFrame: (frame: number) => void,
) {
  if (!animatedHeadPositions) return
  const frameSize = headCount * 3
  const frameCount = animatedHeadPositions.length / frameSize
  const frame = Math.min(frameCount - 1, Math.max(0, Math.floor(currentTime * fps)))
  if (frame === activeFrame) return
  centers.set(animatedHeadPositions.subarray(frame * frameSize, (frame + 1) * frameSize))
  centerAttribute.needsUpdate = true
  setActiveFrame(frame)
}
