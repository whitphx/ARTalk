import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { AnimationMetadata } from '../types'

type GaussianPointRendererProps = {
  metadata: AnimationMetadata
  previewMode: GaussianPreviewMode
  onLoadState: (state: string) => void
}

export type GaussianPreviewMode = 'head' | 'planes' | 'all'

const SPLAT_VERTEX_SHADER = `
attribute float opacity;
attribute float splatScale;
varying vec3 vColor;
varying float vOpacity;

void main() {
  vColor = color;
  vOpacity = opacity;
  vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * viewPosition;
  float depthScale = 1.0 / max(-viewPosition.z, 0.001);
  gl_PointSize = clamp(splatScale * depthScale * 1800.0, 2.0, 58.0);
}
`

const SPLAT_FRAGMENT_SHADER = `
varying vec3 vColor;
varying float vOpacity;

void main() {
  vec2 centered = gl_PointCoord - vec2(0.5);
  float radius2 = dot(centered, centered) * 4.0;
  if (radius2 > 1.0) discard;
  float alpha = exp(-radius2 * 3.2) * vOpacity;
  gl_FragColor = vec4(vColor, alpha);
}
`

export function GaussianPointRenderer({ metadata, previewMode, onLoadState }: GaussianPointRendererProps) {
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
      const [xyzResponse, colorResponse, opacityResponse, scaleResponse] = await Promise.all([
        fetch(metadata.gaussianUrls!.xyz),
        fetch(metadata.gaussianUrls!.colors),
        fetch(metadata.gaussianUrls!.opacities),
        fetch(metadata.gaussianUrls!.scales),
      ])
      if (!xyzResponse.ok) throw new Error(`Failed to load Gaussian positions: ${xyzResponse.status}`)
      if (!colorResponse.ok) throw new Error(`Failed to load Gaussian colors: ${colorResponse.status}`)
      if (!opacityResponse.ok) throw new Error(`Failed to load Gaussian opacities: ${opacityResponse.status}`)
      if (!scaleResponse.ok) throw new Error(`Failed to load Gaussian scales: ${scaleResponse.status}`)

      const [xyzBuffer, colorBuffer, opacityBuffer, scaleBuffer] = await Promise.all([
        xyzResponse.arrayBuffer(),
        colorResponse.arrayBuffer(),
        opacityResponse.arrayBuffer(),
        scaleResponse.arrayBuffer(),
      ])
      if (disposed) return

      const sourcePositions = new Float32Array(xyzBuffer)
      const sourceColors = new Float32Array(colorBuffer)
      const sourceOpacities = new Float32Array(opacityBuffer)
      const sourceScales = new Float32Array(scaleBuffer)
      const count = metadata.gaussianCount!
      const colorChannels = metadata.gaussianColorChannels!
      if (sourcePositions.length !== count * 3) throw new Error('Invalid Gaussian position buffer size')
      if (sourceColors.length !== count * colorChannels) throw new Error('Invalid Gaussian color buffer size')
      if (sourceOpacities.length !== count) throw new Error('Invalid Gaussian opacity buffer size')
      if (sourceScales.length !== count * 3) throw new Error('Invalid Gaussian scale buffer size')

      const range = gaussianPreviewRange(previewMode, count)
      const previewCount = range.end - range.start
      const positions = new Float32Array(previewCount * 3)
      const previewColors = new Float32Array(previewCount * 3)
      const previewOpacities = new Float32Array(previewCount)
      const previewScales = new Float32Array(previewCount)
      for (let index = 0; index < previewCount; index += 1) {
        const sourceIndex = range.start + index
        const sourceOffset = sourceIndex * 3
        positions[index * 3] = sourcePositions[sourceOffset]
        positions[index * 3 + 1] = sourcePositions[sourceOffset + 1]
        positions[index * 3 + 2] = sourcePositions[sourceOffset + 2]
        const colorOffset = sourceIndex * colorChannels
        const opacity = Math.min(Math.max(sourceOpacities[sourceIndex], 0.02), 1)
        const scaleOffset = sourceIndex * 3
        const averageScale =
          (sourceScales[scaleOffset] + sourceScales[scaleOffset + 1] + sourceScales[scaleOffset + 2]) / 3
        const previewOffset = index * 3
        previewColors[previewOffset] = sigmoid(sourceColors[colorOffset])
        previewColors[previewOffset + 1] = sigmoid(sourceColors[colorOffset + 1])
        previewColors[previewOffset + 2] = sigmoid(sourceColors[colorOffset + 2])
        previewOpacities[index] = Math.max(opacity, 0.12)
        previewScales[index] = Math.max(averageScale, 0.006)
      }

      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
      geometry.setAttribute('color', new THREE.BufferAttribute(previewColors, 3))
      geometry.setAttribute('opacity', new THREE.BufferAttribute(previewOpacities, 1))
      geometry.setAttribute('splatScale', new THREE.BufferAttribute(previewScales, 1))
      geometry.computeBoundingSphere()

      const material = new THREE.ShaderMaterial({
        vertexShader: SPLAT_VERTEX_SHADER,
        fragmentShader: SPLAT_FRAGMENT_SHADER,
        vertexColors: true,
        transparent: true,
        blending: THREE.NormalBlending,
        depthWrite: false,
        depthTest: false,
      })
      const points = new THREE.Points(geometry, material)
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
        if (object instanceof THREE.Points) {
          object.geometry.dispose()
          const materials = Array.isArray(object.material) ? object.material : [object.material]
          materials.forEach((material) => material.dispose())
        }
      })
    }
  }, [metadata, previewMode, onLoadState])

  return <canvas ref={canvasRef} aria-label="Experimental Gaussian avatar renderer" />
}

function sigmoid(value: number) {
  return 1 / (1 + Math.exp(-value))
}

function gaussianPreviewRange(previewMode: GaussianPreviewMode, count: number) {
  const headCount = Math.min(5023, count)
  if (previewMode === 'head') return { start: 0, end: headCount }
  if (previewMode === 'planes') return { start: headCount, end: count }
  return { start: 0, end: count }
}
