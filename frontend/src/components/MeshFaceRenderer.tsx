import { useEffect, useRef } from 'react'
import type { RefObject } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { AnimationMetadata } from '../types'

export type MeshMaterialMode = 'skin' | 'debug' | 'normal'

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

  materialModeRef.current = materialMode
  wireframeRef.current = wireframe

  useEffect(() => {
    const mesh = meshRef.current
    if (!mesh) return
    const previousMaterial = mesh.material
    mesh.material = createFaceMaterial(materialMode, wireframe)
    disposeMaterial(previousMaterial)
  }, [materialMode, wireframe])

  useEffect(() => {
    resetCameraFrame(cameraRef.current, controlsRef.current, groupRef.current, boundingSphereRef.current)
  }, [cameraResetSignal])

  useEffect(() => {
    if (!metadata) {
      onLoadState('Waiting for animation data')
      return
    }
    if (!canvasRef.current) return

    const animation = metadata
    let disposed = false
    let animationId = 0
    let resizeFrame = 0
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
      const [vertexResponse, faceResponse] = await Promise.all([
        fetch(animation.verticesUrl),
        fetch(animation.facesUrl),
      ])
      if (!vertexResponse.ok) throw new Error(`Failed to load vertices: ${vertexResponse.status}`)
      if (!faceResponse.ok) throw new Error(`Failed to load faces: ${faceResponse.status}`)

      const [vertexBuffer, faceBuffer] = await Promise.all([
        vertexResponse.arrayBuffer(),
        faceResponse.arrayBuffer(),
      ])
      if (disposed) return

      const allVertices = new Float32Array(vertexBuffer)
      const faces = new Uint32Array(faceBuffer)
      const frameSize = animation.vertexCount * 3
      if (allVertices.length < frameSize || allVertices.length % frameSize !== 0) {
        throw new Error('Invalid vertex buffer size')
      }
      if (faces.length !== animation.faceCount * 3) {
        throw new Error('Invalid face buffer size')
      }

      const positions = new Float32Array(frameSize)
      positions.set(allVertices.subarray(0, frameSize))

      const geometry = new THREE.BufferGeometry()
      geometry.setIndex(new THREE.BufferAttribute(faces, 1))
      const positionAttribute = new THREE.BufferAttribute(positions, 3)
      positionAttribute.setUsage(THREE.DynamicDrawUsage)
      geometry.setAttribute('position', positionAttribute)
      geometry.computeVertexNormals()
      geometry.computeBoundingSphere()

      const material = createFaceMaterial(materialModeRef.current, wireframeRef.current)
      const mesh = new THREE.Mesh(geometry, material)
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
      }

      function scheduleResize() {
        window.cancelAnimationFrame(resizeFrame)
        resizeFrame = window.requestAnimationFrame(resize)
      }

      resizeObserver = new ResizeObserver(scheduleResize)
      resizeObserver.observe(canvas.parentElement ?? canvas)
      resize()

      function renderLoop() {
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
        controls.update()
        renderer.render(scene, camera)
        animationId = window.requestAnimationFrame(renderLoop)
      }

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
  return new THREE.MeshPhysicalMaterial({
    color: 0xd8a58d,
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
