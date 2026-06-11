import { useEffect, useRef } from 'react'
import type { RefObject } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { AnimationMetadata } from '../types'

type GaussianPointRendererProps = {
  metadata: AnimationMetadata
  audioRef: RefObject<HTMLAudioElement | null>
  previewMode: GaussianPreviewMode
  viewMode: GaussianViewMode
  onLoadState: (state: string) => void
}

export type GaussianPreviewMode = 'head' | 'planes' | 'all'
export type GaussianViewMode = 'orbit' | 'gagavatar'

const GAGAVATAR_HEAD_GAUSSIAN_COUNT = 5023
const TEMP_VECTOR_A = new THREE.Vector3()
const TEMP_VECTOR_B = new THREE.Vector3()
const TEMP_MATRIX_A = new THREE.Matrix4()
const TEMP_MATRIX_B = new THREE.Matrix4()
const TEMP_MATRIX_C = new THREE.Matrix4()
const ALL_PREVIEW_HEAD_OPACITY_SCALE = 1.25
const ALL_PREVIEW_PLANE_OPACITY_SCALE = 0.48
const ALL_PREVIEW_HEAD_SCALE_BOOST = 1.12

const SPLAT_VERTEX_SHADER = `
attribute vec3 center;
attribute vec3 gaussianColor;
attribute float gaussianOpacity;
attribute vec3 gaussianScale;
attribute vec4 gaussianRotation;
attribute float gaussianUseMirrorView;
uniform bool useGaussianViewTransform;
uniform mat4 gaussianViewMatrix;
uniform mat4 gaussianMirrorViewMatrix;
varying vec3 vColor;
varying float vOpacity;
varying vec2 vQuad;

vec3 rotateByQuaternion(vec3 value, vec4 quaternion) {
  return value + 2.0 * cross(quaternion.xyz, cross(quaternion.xyz, value) + quaternion.w * value);
}

vec2 eigenVectorFor(float cov00, float cov01, float cov11, float eigenValue) {
  vec2 axis = abs(cov01) > 0.000001
    ? vec2(cov01, eigenValue - cov00)
    : (cov00 >= cov11 ? vec2(1.0, 0.0) : vec2(0.0, 1.0));
  return normalize(axis);
}

void main() {
  vColor = gaussianColor;
  vOpacity = gaussianOpacity;
  vQuad = position.xy;
  mat4 activeGaussianViewMatrix = gaussianUseMirrorView > 0.5 ? gaussianMirrorViewMatrix : gaussianViewMatrix;
  vec3 worldAxisX = rotateByQuaternion(vec3(gaussianScale.x, 0.0, 0.0), gaussianRotation);
  vec3 worldAxisY = rotateByQuaternion(vec3(0.0, gaussianScale.y, 0.0), gaussianRotation);
  vec3 worldAxisZ = rotateByQuaternion(vec3(0.0, 0.0, gaussianScale.z), gaussianRotation);
  vec3 axisX = useGaussianViewTransform ? mat3(activeGaussianViewMatrix) * worldAxisX : mat3(modelViewMatrix) * worldAxisX;
  vec3 axisY = useGaussianViewTransform ? mat3(activeGaussianViewMatrix) * worldAxisY : mat3(modelViewMatrix) * worldAxisY;
  vec3 axisZ = useGaussianViewTransform ? mat3(activeGaussianViewMatrix) * worldAxisZ : mat3(modelViewMatrix) * worldAxisZ;
  float cov00 = dot(vec3(axisX.x, axisY.x, axisZ.x), vec3(axisX.x, axisY.x, axisZ.x));
  float cov01 = dot(vec3(axisX.x, axisY.x, axisZ.x), vec3(axisX.y, axisY.y, axisZ.y));
  float cov11 = dot(vec3(axisX.y, axisY.y, axisZ.y), vec3(axisX.y, axisY.y, axisZ.y));
  float trace = cov00 + cov11;
  float discriminant = sqrt(max((cov00 - cov11) * (cov00 - cov11) + 4.0 * cov01 * cov01, 0.0));
  float lambda0 = max((trace + discriminant) * 0.5, 0.000036);
  float lambda1 = max((trace - discriminant) * 0.5, 0.000036);
  vec2 majorDirection = eigenVectorFor(cov00, cov01, cov11, lambda0);
  vec2 majorAxis = majorDirection * sqrt(lambda0);
  vec2 minorAxis = vec2(-majorDirection.y, majorDirection.x) * sqrt(lambda1);
  vec2 viewOffset = (majorAxis * position.x + minorAxis * position.y) * 2.6;
  vec4 viewPosition = useGaussianViewTransform ? activeGaussianViewMatrix * vec4(center, 1.0) : modelViewMatrix * vec4(center, 1.0);
  viewPosition.xy += viewOffset;
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
  viewMode,
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
    let activeHeadSample = -1
    let activeTransformFrame = -1
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
    controls.enabled = viewMode === 'orbit'
    controls.minDistance = 0.35
    controls.maxDistance = 7.5

    async function load() {
      onLoadState('Loading Gaussian buffers')
      const shouldLoadAnimatedHead = previewMode !== 'planes' && Boolean(metadata.gaussianUrls!.headXyz)
      const [xyzResponse, headXyzResponse, transformResponse, colorResponse, opacityResponse, scaleResponse, rotationResponse] = await Promise.all([
        fetch(metadata.gaussianUrls!.xyz),
        shouldLoadAnimatedHead ? fetch(metadata.gaussianUrls!.headXyz!) : Promise.resolve(null),
        metadata.gaussianUrls!.transforms ? fetch(metadata.gaussianUrls!.transforms) : Promise.resolve(null),
        fetch(metadata.gaussianUrls!.colors),
        fetch(metadata.gaussianUrls!.opacities),
        fetch(metadata.gaussianUrls!.scales),
        fetch(metadata.gaussianUrls!.rotations),
      ])
      if (!xyzResponse.ok) throw new Error(`Failed to load Gaussian positions: ${xyzResponse.status}`)
      if (headXyzResponse && !headXyzResponse.ok) {
        throw new Error(`Failed to load animated Gaussian head positions: ${headXyzResponse.status}`)
      }
      if (transformResponse && !transformResponse.ok) {
        throw new Error(`Failed to load Gaussian transforms: ${transformResponse.status}`)
      }
      if (!colorResponse.ok) throw new Error(`Failed to load Gaussian colors: ${colorResponse.status}`)
      if (!opacityResponse.ok) throw new Error(`Failed to load Gaussian opacities: ${opacityResponse.status}`)
      if (!scaleResponse.ok) throw new Error(`Failed to load Gaussian scales: ${scaleResponse.status}`)
      if (!rotationResponse.ok) throw new Error(`Failed to load Gaussian rotations: ${rotationResponse.status}`)

      const [xyzBuffer, headXyzBuffer, transformBuffer, colorBuffer, opacityBuffer, scaleBuffer, rotationBuffer] = await Promise.all([
        xyzResponse.arrayBuffer(),
        headXyzResponse ? headXyzResponse.arrayBuffer() : Promise.resolve(null),
        transformResponse ? transformResponse.arrayBuffer() : Promise.resolve(null),
        colorResponse.arrayBuffer(),
        opacityResponse.arrayBuffer(),
        scaleResponse.arrayBuffer(),
        rotationResponse.arrayBuffer(),
      ])
      if (disposed) return

      const sourcePositions = new Float32Array(xyzBuffer)
      const animatedHeadPositions = headXyzBuffer ? new Float32Array(headXyzBuffer) : null
      const gaussianTransforms = transformBuffer ? new Float32Array(transformBuffer) : null
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
      if (
        gaussianTransforms &&
        gaussianTransforms.length !== (metadata.gaussianTransformFrameCount ?? metadata.frameCount) * 12
      ) {
        throw new Error('Invalid Gaussian transform buffer size')
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
      const previewMirrorViewFlags = new Float32Array(previewCount)
      const sortedCenters = new Float32Array(previewCount * 3)
      const sortedColors = new Float32Array(previewCount * 3)
      const sortedOpacities = new Float32Array(previewCount)
      const sortedScales = new Float32Array(previewCount * 3)
      const sortedRotations = new Float32Array(previewCount * 4)
      const sortedMirrorViewFlags = new Float32Array(previewCount)
      const sortOrder = Array.from({ length: previewCount }, (_, index) => index)
      const depthValues = new Float32Array(previewCount)
      for (let index = 0; index < previewCount; index += 1) {
        const sourceIndex = range.start + index
        const sourceOffset = sourceIndex * 3
        const isHeadGaussian = sourceIndex < headCount
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
        previewOpacities[index] = weightedOpacity(opacity, previewMode, isHeadGaussian)
        const scaleBoost = previewMode === 'all' && isHeadGaussian ? ALL_PREVIEW_HEAD_SCALE_BOOST : 1
        previewScales[previewOffset] = Math.max(sourceScales[scaleOffset] * scaleBoost, 0.006)
        previewScales[previewOffset + 1] = Math.max(sourceScales[scaleOffset + 1] * scaleBoost, 0.006)
        previewScales[previewOffset + 2] = Math.max(sourceScales[scaleOffset + 2] * scaleBoost, 0.006)
        const rotationOffset = sourceIndex * 4
        const previewRotationOffset = index * 4
        previewRotations[previewRotationOffset] = sourceRotations[rotationOffset]
        previewRotations[previewRotationOffset + 1] = sourceRotations[rotationOffset + 1]
        previewRotations[previewRotationOffset + 2] = sourceRotations[rotationOffset + 2]
        previewRotations[previewRotationOffset + 3] = sourceRotations[rotationOffset + 3]
        previewMirrorViewFlags[index] = isHeadGaussian ? 1 : 0
      }

      const geometry = new THREE.InstancedBufferGeometry()
      geometry.instanceCount = previewCount
      geometry.setAttribute('position', new THREE.BufferAttribute(buildUnitQuadVertices(), 3))
      geometry.setAttribute('center', new THREE.InstancedBufferAttribute(sortedCenters, 3).setUsage(THREE.DynamicDrawUsage))
      geometry.setAttribute(
        'gaussianColor',
        new THREE.InstancedBufferAttribute(sortedColors, 3).setUsage(THREE.DynamicDrawUsage),
      )
      geometry.setAttribute(
        'gaussianOpacity',
        new THREE.InstancedBufferAttribute(sortedOpacities, 1).setUsage(THREE.DynamicDrawUsage),
      )
      geometry.setAttribute('gaussianScale', new THREE.InstancedBufferAttribute(sortedScales, 3).setUsage(THREE.DynamicDrawUsage))
      geometry.setAttribute(
        'gaussianRotation',
        new THREE.InstancedBufferAttribute(sortedRotations, 4).setUsage(THREE.DynamicDrawUsage),
      )
      geometry.setAttribute(
        'gaussianUseMirrorView',
        new THREE.InstancedBufferAttribute(sortedMirrorViewFlags, 1).setUsage(THREE.DynamicDrawUsage),
      )
      geometry.boundingSphere = boundingSphereForCenters(centers)

      const material = new THREE.ShaderMaterial({
        vertexShader: SPLAT_VERTEX_SHADER,
        fragmentShader: SPLAT_FRAGMENT_SHADER,
        uniforms: {
          useGaussianViewTransform: { value: viewMode === 'gagavatar' },
          gaussianViewMatrix: { value: new THREE.Matrix4() },
          gaussianMirrorViewMatrix: { value: new THREE.Matrix4() },
        },
        transparent: true,
        blending: THREE.NormalBlending,
        depthWrite: false,
        depthTest: false,
      })
      const points = new THREE.Mesh(geometry, material)
      points.frustumCulled = false
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
        camera.fov = viewMode === 'gagavatar' && metadata.gaussianCamera
          ? THREE.MathUtils.radToDeg(2 * Math.atan(1 / metadata.gaussianCamera.focalY))
          : 28
        camera.aspect = width / Math.max(height, 1)
        camera.updateProjectionMatrix()
      }

      function scheduleResize() {
        window.cancelAnimationFrame(resizeFrame)
        resizeFrame = window.requestAnimationFrame(resize)
      }

      let sortRequested = true
      function renderLoop() {
        if (disposed) return
        const headFrameChanged = updateAnimatedHeadCenters(
          centers,
          animatedHeadPositions,
          audioRef.current?.currentTime ?? 0,
          metadata.fps,
          headCount,
          activeHeadSample,
          (sample) => {
            activeHeadSample = sample
          },
        )
        const transformFrameChanged = updateGaussianViewTransform(
          material.uniforms.gaussianViewMatrix.value as THREE.Matrix4,
          material.uniforms.gaussianMirrorViewMatrix.value as THREE.Matrix4,
          gaussianTransforms,
          audioRef.current?.currentTime ?? 0,
          metadata.fps,
          viewMode,
          activeTransformFrame,
          (frame) => {
            activeTransformFrame = frame
          },
        )
        sortRequested = sortRequested || headFrameChanged || transformFrameChanged || controls.update()
        if (sortRequested) {
          sortGaussianInstances(
            {
              centers,
              colors: previewColors,
              opacities: previewOpacities,
              scales: previewScales,
              rotations: previewRotations,
              mirrorViewFlags: previewMirrorViewFlags,
            },
            {
              centers: sortedCenters,
              colors: sortedColors,
              opacities: sortedOpacities,
              scales: sortedScales,
              rotations: sortedRotations,
              mirrorViewFlags: sortedMirrorViewFlags,
            },
            geometry,
            sortOrder,
            depthValues,
            camera,
            points.position,
            viewMode === 'gagavatar'
              ? {
                  native: material.uniforms.gaussianViewMatrix.value as THREE.Matrix4,
                  mirror: material.uniforms.gaussianMirrorViewMatrix.value as THREE.Matrix4,
                }
              : null,
          )
          sortRequested = false
        }
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
  }, [metadata, audioRef, previewMode, viewMode, onLoadState])

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

function weightedOpacity(opacity: number, previewMode: GaussianPreviewMode, isHeadGaussian: boolean) {
  const baseOpacity = Math.max(opacity, 0.12)
  if (previewMode !== 'all') return baseOpacity
  const scale = isHeadGaussian ? ALL_PREVIEW_HEAD_OPACITY_SCALE : ALL_PREVIEW_PLANE_OPACITY_SCALE
  return Math.min(baseOpacity * scale, 1)
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
  animatedHeadPositions: Float32Array | null,
  currentTime: number,
  fps: number,
  headCount: number,
  activeSample: number,
  setActiveSample: (sample: number) => void,
) {
  if (!animatedHeadPositions) return false
  const frameSize = headCount * 3
  const frameCount = animatedHeadPositions.length / frameSize
  const framePosition = Math.min(frameCount - 1, Math.max(0, currentTime * fps))
  const sample = Math.round(framePosition * 1000)
  if (sample === activeSample) return false
  const frame = Math.floor(framePosition)
  const nextFrame = Math.min(frameCount - 1, frame + 1)
  const mix = framePosition - frame
  const frameOffset = frame * frameSize
  const nextFrameOffset = nextFrame * frameSize
  if (mix === 0) {
    for (let index = 0; index < frameSize; index += 1) {
      centers[index] = animatedHeadPositions[frameOffset + index]
    }
  } else {
    for (let index = 0; index < frameSize; index += 1) {
      const value = animatedHeadPositions[frameOffset + index]
      centers[index] = value + (animatedHeadPositions[nextFrameOffset + index] - value) * mix
    }
  }
  setActiveSample(sample)
  return true
}

function updateGaussianViewTransform(
  nativeMatrix: THREE.Matrix4,
  mirrorMatrix: THREE.Matrix4,
  transforms: Float32Array | null,
  currentTime: number,
  fps: number,
  viewMode: GaussianViewMode,
  activeFrame: number,
  setActiveFrame: (frame: number) => void,
) {
  if (viewMode !== 'gagavatar' || !transforms) return false
  const frameSize = 12
  const frameCount = transforms.length / frameSize
  const frame = Math.min(frameCount - 1, Math.max(0, Math.floor(currentTime * fps)))
  if (frame === activeFrame) return false
  writeGaussianViewMatrix(nativeMatrix, transforms, frame, false)
  invertViewDelta(nativeMatrix, transforms, false)
  writeGaussianViewMatrix(mirrorMatrix, transforms, frame, true)
  invertViewDelta(mirrorMatrix, transforms, true)
  setActiveFrame(frame)
  return true
}

function writeGaussianViewMatrix(matrix: THREE.Matrix4, transforms: Float32Array, frame: number, mirrorViewX: boolean) {
  const offset = frame * 12
  const r00 = transforms[offset]
  const r01 = transforms[offset + 1]
  const r02 = transforms[offset + 2]
  const tx = transforms[offset + 3]
  const r10 = transforms[offset + 4]
  const r11 = transforms[offset + 5]
  const r12 = transforms[offset + 6]
  const ty = transforms[offset + 7]
  const r20 = transforms[offset + 8]
  const r21 = transforms[offset + 9]
  const r22 = transforms[offset + 10]
  const tz = transforms[offset + 11]
  const xSign = mirrorViewX ? 1 : -1
  matrix.set(
    xSign * r00, xSign * r01, xSign * r02, xSign * tx,
    r10, r11, r12, ty,
    -r20, -r21, -r22, -tz,
    0, 0, 0, 1,
  )
}

function invertViewDelta(matrix: THREE.Matrix4, transforms: Float32Array, mirrorViewX: boolean) {
  TEMP_MATRIX_A.copy(matrix)
  writeGaussianViewMatrix(TEMP_MATRIX_B, transforms, 0, mirrorViewX)
  TEMP_MATRIX_C.copy(TEMP_MATRIX_A).invert()
  matrix.multiplyMatrices(TEMP_MATRIX_B, TEMP_MATRIX_C)
  matrix.multiply(TEMP_MATRIX_B)
}

type GaussianInstanceAttributes = {
  centers: Float32Array
  colors: Float32Array
  opacities: Float32Array
  scales: Float32Array
  rotations: Float32Array
  mirrorViewFlags: Float32Array
}

type GaussianViewMatrices = {
  native: THREE.Matrix4
  mirror: THREE.Matrix4
}

function sortGaussianInstances(
  source: GaussianInstanceAttributes,
  target: GaussianInstanceAttributes,
  geometry: THREE.InstancedBufferGeometry,
  sortOrder: number[],
  depthValues: Float32Array,
  camera: THREE.Camera,
  objectOffset: THREE.Vector3,
  gaussianViewMatrices: GaussianViewMatrices | null,
) {
  const cameraPosition = TEMP_VECTOR_A
  const cameraDirection = TEMP_VECTOR_B
  if (!gaussianViewMatrices) {
    camera.getWorldPosition(cameraPosition)
    camera.getWorldDirection(cameraDirection)
  }

  for (let index = 0; index < sortOrder.length; index += 1) {
    sortOrder[index] = index
    const offset = index * 3
    const worldX = source.centers[offset] + objectOffset.x
    const worldY = source.centers[offset + 1] + objectOffset.y
    const worldZ = source.centers[offset + 2] + objectOffset.z
    if (gaussianViewMatrices) {
      const gaussianViewMatrix =
        source.mirrorViewFlags[index] > 0.5 ? gaussianViewMatrices.mirror : gaussianViewMatrices.native
      cameraPosition.set(source.centers[offset], source.centers[offset + 1], source.centers[offset + 2])
      cameraPosition.applyMatrix4(gaussianViewMatrix)
      depthValues[index] = -cameraPosition.z
    } else {
      depthValues[index] =
        (worldX - cameraPosition.x) * cameraDirection.x +
        (worldY - cameraPosition.y) * cameraDirection.y +
        (worldZ - cameraPosition.z) * cameraDirection.z
    }
  }
  sortOrder.sort((left, right) => depthValues[right] - depthValues[left])

  for (let targetIndex = 0; targetIndex < sortOrder.length; targetIndex += 1) {
    const sourceIndex = sortOrder[targetIndex]
    copyVector3(source.centers, target.centers, sourceIndex, targetIndex)
    copyVector3(source.colors, target.colors, sourceIndex, targetIndex)
    target.opacities[targetIndex] = source.opacities[sourceIndex]
    copyVector3(source.scales, target.scales, sourceIndex, targetIndex)
    copyVector4(source.rotations, target.rotations, sourceIndex, targetIndex)
    target.mirrorViewFlags[targetIndex] = source.mirrorViewFlags[sourceIndex]
  }

  markAttributeUpdated(geometry, 'center')
  markAttributeUpdated(geometry, 'gaussianColor')
  markAttributeUpdated(geometry, 'gaussianOpacity')
  markAttributeUpdated(geometry, 'gaussianScale')
  markAttributeUpdated(geometry, 'gaussianRotation')
  markAttributeUpdated(geometry, 'gaussianUseMirrorView')
}

function copyVector3(source: Float32Array, target: Float32Array, sourceIndex: number, targetIndex: number) {
  const sourceOffset = sourceIndex * 3
  const targetOffset = targetIndex * 3
  target[targetOffset] = source[sourceOffset]
  target[targetOffset + 1] = source[sourceOffset + 1]
  target[targetOffset + 2] = source[sourceOffset + 2]
}

function copyVector4(source: Float32Array, target: Float32Array, sourceIndex: number, targetIndex: number) {
  const sourceOffset = sourceIndex * 4
  const targetOffset = targetIndex * 4
  target[targetOffset] = source[sourceOffset]
  target[targetOffset + 1] = source[sourceOffset + 1]
  target[targetOffset + 2] = source[sourceOffset + 2]
  target[targetOffset + 3] = source[sourceOffset + 3]
}

function markAttributeUpdated(geometry: THREE.BufferGeometry, name: string) {
  const attribute = geometry.getAttribute(name)
  if (attribute) attribute.needsUpdate = true
}
