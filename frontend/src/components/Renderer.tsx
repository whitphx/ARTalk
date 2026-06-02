import { useEffect, useRef, useState } from 'react'
import { Loader2, Radio } from 'lucide-react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { AnimationMetadata } from '../types'

type RendererProps = {
  metadata: AnimationMetadata | null
}

export function Renderer({ metadata }: RendererProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [loadState, setLoadState] = useState('Waiting for animation data')

  useEffect(() => {
    if (!metadata || !canvasRef.current) return

    const animation = metadata
    let disposed = false
    let animationId = 0
    let resizeObserver: ResizeObserver | null = null
    const canvas = canvasRef.current
    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
    })
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(28, 1, 0.01, 100)
    const controls = new OrbitControls(camera, canvas)
    const group = new THREE.Group()
    let activeFrame = -1

    controls.enableDamping = true
    controls.enablePan = false
    controls.minDistance = 0.35
    controls.maxDistance = 3.5
    scene.add(group)
    scene.add(new THREE.HemisphereLight(0xe8f3ff, 0x39402e, 2.2))

    const key = new THREE.DirectionalLight(0xffffff, 2.8)
    key.position.set(1.2, 1.6, 2.4)
    scene.add(key)

    const rim = new THREE.DirectionalLight(0x73f4c8, 1.5)
    rim.position.set(-1.4, 0.3, 1.8)
    scene.add(rim)

    async function load() {
      setLoadState('Loading mesh buffers')
      const [vertexBuffer, faceBuffer] = await Promise.all([
        fetch(animation.verticesUrl).then((res) => res.arrayBuffer()),
        fetch(animation.facesUrl).then((res) => res.arrayBuffer()),
      ])
      if (disposed) return

      const allVertices = new Float32Array(vertexBuffer)
      const faces = new Uint32Array(faceBuffer)
      const frameSize = animation.vertexCount * 3
      const positions = new Float32Array(frameSize)
      positions.set(allVertices.subarray(0, frameSize))

      const geometry = new THREE.BufferGeometry()
      geometry.setIndex(new THREE.BufferAttribute(faces, 1))
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
      geometry.computeVertexNormals()
      geometry.computeBoundingSphere()

      const material = new THREE.MeshStandardMaterial({
        color: 0x8eb3f7,
        roughness: 0.42,
        metalness: 0.05,
      })
      const mesh = new THREE.Mesh(geometry, material)
      group.add(mesh)

      const sphere = geometry.boundingSphere
      if (sphere) {
        group.position.set(-sphere.center.x, -sphere.center.y, -sphere.center.z)
        camera.position.set(0, 0.02, Math.max(sphere.radius * 4.5, 0.7))
        controls.target.set(0, 0.01, 0)
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

      resizeObserver = new ResizeObserver(resize)
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

      setLoadState('Ready')
      renderLoop()
    }

    load().catch((error: unknown) => {
      if (!disposed) setLoadState(error instanceof Error ? error.message : String(error))
    })

    return () => {
      disposed = true
      window.cancelAnimationFrame(animationId)
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
    }
  }, [metadata])

  return (
    <section className="stage" aria-label="Generated avatar preview">
      <div className="viewport">
        <canvas ref={canvasRef} aria-label="3D avatar renderer" />
        {!metadata && (
          <div className="empty-state">
            <Radio aria-hidden="true" />
            <span>No animation loaded.</span>
          </div>
        )}
        {metadata && loadState !== 'Ready' && (
          <div className="loading-state">
            <Loader2 aria-hidden="true" />
            <span>{loadState}</span>
          </div>
        )}
      </div>
      <div className="transport">
        <audio ref={audioRef} src={metadata?.audioUrl} controls />
        <div className="readout" aria-live="polite">
          {metadata
            ? `${metadata.frameCount} frames · ${metadata.vertexCount} vertices · ${metadata.fps} fps`
            : 'No render loaded'}
        </div>
      </div>
    </section>
  )
}
