import { useEffect, useRef } from 'react'
import type { RefObject } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { AnimationMetadata } from '../types'

export type MeshMaterialMode = 'skin' | 'region' | 'debug' | 'normal'

const REGION_COLORS = [
  new THREE.Color(0xd8a58d),
  new THREE.Color(0xa85858),
  new THREE.Color(0x2a1717),
  new THREE.Color(0xe8e0d4),
]
const SKIN_COLORS = [
  new THREE.Color(0xd9a38a),
  new THREE.Color(0xbf6566),
  new THREE.Color(0x4a2223),
  new THREE.Color(0xeadbcf),
]

type MeshFaceRendererProps = {
  metadata: AnimationMetadata | null
  audioRef: RefObject<HTMLAudioElement | null>
  materialMode: MeshMaterialMode
  wireframe: boolean
  cameraResetSignal: number
  onLoadState: (state: string) => void
}

export function MeshFaceRenderer({
  metadata,
  audioRef,
  materialMode,
  wireframe,
  cameraResetSignal,
  onLoadState,
}: MeshFaceRendererProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const meshRef = useRef<THREE.Mesh | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const controlsRef = useRef<OrbitControls | null>(null)
  const groupRef = useRef<THREE.Group | null>(null)
  const boundingSphereRef = useRef<THREE.Sphere | null>(null)
  const materialModeRef = useRef(materialMode)
  const wireframeRef = useRef(wireframe)
  const requestRenderRef = useRef<(() => void) | null>(null)

  materialModeRef.current = materialMode
  wireframeRef.current = wireframe

  useEffect(() => {
    const mesh = meshRef.current
    if (!mesh) return
    const previousMaterial = mesh.material
    applyColorAttribute(mesh, materialMode)
    mesh.material = createFaceMaterial(materialMode, wireframe)
    disposeMaterial(previousMaterial)
    requestRenderRef.current?.()
  }, [materialMode, wireframe])

  useEffect(() => {
    resetCameraFrame(cameraRef.current, controlsRef.current, groupRef.current, boundingSphereRef.current)
    requestRenderRef.current?.()
  }, [cameraResetSignal])

  useEffect(() => {
    if (!metadata) {
      onLoadState('Waiting for animation data')
      return
    }
    if (!canvasRef.current) return

    const animation = metadata
    let disposed = false
    let renderFrameId = 0
    let resizeFrame = 0
    let renderQueued = false
    let resizeObserver: ResizeObserver | null = null
    const canvas = canvasRef.current
    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
    })
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.05

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(28, 1, 0.01, 100)
    const controls = new OrbitControls(camera, canvas)
    const group = new THREE.Group()
    let activeFrame = -1
    cameraRef.current = camera
    controlsRef.current = controls
    groupRef.current = group

    controls.enableDamping = true
    controls.enablePan = false
    controls.minDistance = 0.35
    controls.maxDistance = 3.5
    scene.add(group)

    scene.add(new THREE.HemisphereLight(0xfff2df, 0x26342f, 2.0))

    const key = new THREE.DirectionalLight(0xfff5ea, 2.6)
    key.position.set(1.2, 1.7, 2.6)
    scene.add(key)

    const fill = new THREE.DirectionalLight(0x94d8ff, 0.8)
    fill.position.set(-1.5, 0.4, 1.5)
    scene.add(fill)

    const rim = new THREE.DirectionalLight(0xffffff, 1.4)
    rim.position.set(-0.8, 1.2, -1.4)
    scene.add(rim)

    async function load() {
      onLoadState('Loading mesh buffers')
      const [vertexResponse, faceResponse, regionResponse] = await Promise.all([
        fetch(animation.verticesUrl),
        fetch(animation.facesUrl),
        animation.regionLabelsUrl ? fetch(animation.regionLabelsUrl) : Promise.resolve(null),
      ])
      if (!vertexResponse.ok) throw new Error(`Failed to load vertices: ${vertexResponse.status}`)
      if (!faceResponse.ok) throw new Error(`Failed to load faces: ${faceResponse.status}`)
      if (regionResponse && !regionResponse.ok) {
        throw new Error(`Failed to load region labels: ${regionResponse.status}`)
      }

      const [vertexBuffer, faceBuffer, regionBuffer] = await Promise.all([
        vertexResponse.arrayBuffer(),
        faceResponse.arrayBuffer(),
        regionResponse ? regionResponse.arrayBuffer() : Promise.resolve(null),
      ])
      if (disposed) return

      const allVertices = new Float32Array(vertexBuffer)
      const faces = new Uint32Array(faceBuffer)
      const regionLabels = regionBuffer ? new Uint8Array(regionBuffer) : null
      const frameSize = animation.vertexCount * 3
      if (allVertices.length < frameSize || allVertices.length % frameSize !== 0) {
        throw new Error('Invalid vertex buffer size')
      }
      if (faces.length !== animation.faceCount * 3) {
        throw new Error('Invalid face buffer size')
      }
      if (regionLabels && regionLabels.length !== animation.vertexCount) {
        throw new Error('Invalid region label buffer size')
      }

      const positions = new Float32Array(frameSize)
      positions.set(allVertices.subarray(0, frameSize))

      const geometry = new THREE.BufferGeometry()
      geometry.setIndex(new THREE.BufferAttribute(faces, 1))
      const positionAttribute = new THREE.BufferAttribute(positions, 3)
      positionAttribute.setUsage(THREE.DynamicDrawUsage)
      geometry.setAttribute('position', positionAttribute)
      geometry.userData.skinColors = buildMeshColors(positions, regionLabels, 'skin')
      geometry.userData.regionColors = buildMeshColors(positions, regionLabels, 'region')
      geometry.setAttribute('color', geometry.userData.skinColors)
      geometry.computeVertexNormals()
      geometry.computeBoundingSphere()

      const material = createFaceMaterial(materialModeRef.current, wireframeRef.current)
      const mesh = new THREE.Mesh(geometry, material)
      applyColorAttribute(mesh, materialModeRef.current)
      meshRef.current = mesh
      group.add(mesh)

      const sphere = geometry.boundingSphere
      boundingSphereRef.current = sphere ? sphere.clone() : null
      resetCameraFrame(camera, controls, group, boundingSphereRef.current)

      function resize() {
        const parent = canvas.parentElement
        if (!parent) return
        const { width, height } = parent.getBoundingClientRect()
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
        renderer.setSize(width, height, false)
        camera.aspect = width / Math.max(height, 1)
        camera.updateProjectionMatrix()
        requestRender()
      }

      function scheduleResize() {
        window.cancelAnimationFrame(resizeFrame)
        resizeFrame = window.requestAnimationFrame(resize)
      }

      resizeObserver = new ResizeObserver(scheduleResize)
      resizeObserver.observe(canvas.parentElement ?? canvas)
      resize()

      function requestRender() {
        if (disposed || renderQueued) return
        renderQueued = true
        renderFrameId = window.requestAnimationFrame(renderFrame)
      }

      function updateFrame() {
        if (disposed) return
        const audio = audioRef.current
        const frame = audio
          ? Math.min(animation.frameCount - 1, Math.floor(audio.currentTime * animation.fps))
          : 0
        if (frame !== activeFrame) {
          activeFrame = frame
          const offset = frame * frameSize
          positions.set(allVertices.subarray(offset, offset + frameSize))
          geometry.attributes.position.needsUpdate = true
          geometry.computeVertexNormals()
        }
      }

      function renderFrame() {
        if (disposed) return
        renderQueued = false
        updateFrame()
        const controlsChanged = controls.update()
        renderer.render(scene, camera)
        const audio = audioRef.current
        if (audio && !audio.paused && !audio.ended) {
          requestRender()
        } else if (controlsChanged) {
          requestRender()
        }
      }

      const audio = audioRef.current
      const requestRenderOnAudioChange = () => requestRender()
      audio?.addEventListener('play', requestRenderOnAudioChange)
      audio?.addEventListener('seeked', requestRenderOnAudioChange)
      audio?.addEventListener('timeupdate', requestRenderOnAudioChange)
      controls.addEventListener('change', requestRender)
      requestRenderRef.current = requestRender
      onLoadState('Ready')
      requestRender()

      return () => {
        audio?.removeEventListener('play', requestRenderOnAudioChange)
        audio?.removeEventListener('seeked', requestRenderOnAudioChange)
        audio?.removeEventListener('timeupdate', requestRenderOnAudioChange)
        controls.removeEventListener('change', requestRender)
      }
    }

    let removeRenderEventListeners: (() => void) | undefined
    load()
      .then((cleanup) => {
        removeRenderEventListeners = cleanup
      })
      .catch((error: unknown) => {
        if (!disposed) onLoadState(error instanceof Error ? error.message : String(error))
      })

    return () => {
      disposed = true
      removeRenderEventListeners?.()
      requestRenderRef.current = null
      window.cancelAnimationFrame(renderFrameId)
      window.cancelAnimationFrame(resizeFrame)
      resizeObserver?.disconnect()
      renderer.dispose()
      controls.dispose()
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh) {
          object.geometry.dispose()
          const materials = Array.isArray(object.material) ? object.material : [object.material]
          materials.forEach((material) => material.dispose())
        }
      })
      meshRef.current = null
      cameraRef.current = null
      controlsRef.current = null
      groupRef.current = null
      boundingSphereRef.current = null
    }
  }, [metadata, audioRef, onLoadState])

  return <canvas ref={canvasRef} aria-label="3D avatar renderer" />
}

function disposeMaterial(material: THREE.Material | THREE.Material[]) {
  const materials = Array.isArray(material) ? material : [material]
  materials.forEach((item) => item.dispose())
}

function createFaceMaterial(materialMode: MeshMaterialMode, wireframe: boolean) {
  if (materialMode === 'normal') {
    return new THREE.MeshNormalMaterial({ wireframe })
  }
  if (materialMode === 'debug') {
    return new THREE.MeshStandardMaterial({
      color: 0x8eb3f7,
      roughness: 0.42,
      metalness: 0.05,
      wireframe,
    })
  }
  if (materialMode === 'region') {
    return new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.54,
      metalness: 0.0,
      wireframe,
    })
  }
  return new THREE.MeshPhysicalMaterial({
    vertexColors: true,
    roughness: 0.58,
    metalness: 0.0,
    sheen: 0.25,
    sheenRoughness: 0.9,
    wireframe,
  })
}

function resetCameraFrame(
  camera: THREE.PerspectiveCamera | null,
  controls: OrbitControls | null,
  group: THREE.Group | null,
  sphere: THREE.Sphere | null,
) {
  if (!camera || !controls || !group || !sphere) return
  group.position.set(-sphere.center.x, -sphere.center.y, -sphere.center.z)
  camera.position.set(0, 0.02, Math.max(sphere.radius * 4.1, 0.7))
  controls.target.set(0, 0.01, 0)
  controls.update()
}

function buildMeshColors(
  positions: Float32Array,
  regionLabels: Uint8Array | null,
  palette: 'skin' | 'region',
) {
  const vertexCount = positions.length / 3
  const colors = new Float32Array(vertexCount * 3)
  if (regionLabels) {
    for (let index = 0; index < vertexCount; index += 1) {
      writeRegionColor(colors, index, regionLabels[index], palette)
    }
    return new THREE.BufferAttribute(colors, 3)
  }

  const min = new THREE.Vector3(Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY)
  const max = new THREE.Vector3(Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY)
  for (let index = 0; index < vertexCount; index += 1) {
    const x = positions[index * 3]
    const y = positions[index * 3 + 1]
    const z = positions[index * 3 + 2]
    min.x = Math.min(min.x, x)
    min.y = Math.min(min.y, y)
    min.z = Math.min(min.z, z)
    max.x = Math.max(max.x, x)
    max.y = Math.max(max.y, y)
    max.z = Math.max(max.z, z)
  }

  const size = max.sub(min)
  for (let index = 0; index < vertexCount; index += 1) {
    const x = positions[index * 3]
    const y = positions[index * 3 + 1]
    const z = positions[index * 3 + 2]
    const nx = normalizeAxis(x, min.x, size.x)
    const ny = normalizeAxis(y, min.y, size.y)
    const nz = normalizeAxis(z, min.z, size.z)
    const color = colorForRegionLabel(classifyApproximateFaceRegion(nx, ny, nz), palette)
    const base = index * 3
    colors[base] = color.r
    colors[base + 1] = color.g
    colors[base + 2] = color.b
  }

  function classifyApproximateFaceRegion(nx: number, ny: number, nz: number) {
    const centeredX = Math.abs(nx - 0.5)
    if (nz > 0.54 && ny > 0.36 && ny < 0.5 && centeredX < 0.18) return 1
    if (nz > 0.56 && ny > 0.31 && ny <= 0.38 && centeredX < 0.1) return 2
    if (nz > 0.5 && ny > 0.56 && ny < 0.68 && centeredX > 0.13 && centeredX < 0.29) return 3
    return 0
  }

  return new THREE.BufferAttribute(colors, 3)
}

function writeRegionColor(
  colors: Float32Array,
  index: number,
  label: number,
  palette: 'skin' | 'region',
) {
  const color = colorForRegionLabel(label, palette)
  const base = index * 3
  colors[base] = color.r
  colors[base + 1] = color.g
  colors[base + 2] = color.b
}

function colorForRegionLabel(label: number, palette: 'skin' | 'region') {
  const colors = palette === 'skin' ? SKIN_COLORS : REGION_COLORS
  return colors[label] ?? colors[0]
}

function applyColorAttribute(mesh: THREE.Mesh, materialMode: MeshMaterialMode) {
  const geometry = mesh.geometry
  const colors =
    materialMode === 'region'
      ? geometry.userData.regionColors
      : geometry.userData.skinColors
  if (colors instanceof THREE.BufferAttribute) {
    geometry.setAttribute('color', colors)
    colors.needsUpdate = true
  }
}

function normalizeAxis(value: number, min: number, size: number) {
  return size > 0 ? (value - min) / size : 0.5
}
