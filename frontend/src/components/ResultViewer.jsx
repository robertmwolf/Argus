import { useEffect, useRef, useState } from 'react'

const OBB_COLOUR = '#00DCFF'
const HIGHLIGHT_COLOUR = '#FF6B35'

/**
 * Draw one oriented bounding box on the canvas context.
 *
 * @param {CanvasRenderingContext2D} ctx
 * @param {object} obb  - {cx, cy, w, h, angle_deg}
 * @param {number} conf - confidence in [0,1]
 * @param {boolean} highlighted
 * @param {number} scaleX - canvas-width / image-width
 * @param {number} scaleY - canvas-height / image-height
 */
function drawOBB(ctx, obb, conf, highlighted, scaleX, scaleY) {
  const { cx, cy, w, h, angle_deg } = obb
  if ([cx, cy, w, h, angle_deg].some((v) => v == null)) return

  const rad = (angle_deg * Math.PI) / 180

  ctx.save()
  ctx.translate(cx * scaleX, cy * scaleY)
  ctx.rotate(rad)

  ctx.strokeStyle = highlighted ? HIGHLIGHT_COLOUR : OBB_COLOUR
  ctx.lineWidth = highlighted ? 4 : 2
  ctx.globalAlpha = highlighted ? 1.0 : 0.3 + conf * 0.7
  ctx.strokeRect((-w / 2) * scaleX, (-h / 2) * scaleY, w * scaleX, h * scaleY)

  ctx.restore()
}

/**
 * ResultViewer — canvas overlay of detection OBBs on the result image.
 *
 * Props:
 *   jobId            — UUID of the completed job
 *   detections       — array of detection dicts from /api/result
 *   highlightIndex   — index of the detection row to highlight (or null)
 *   onHover(index)   — called when mouse enters an OBB (null = leave)
 */
export default function ResultViewer({ jobId, detections, highlightIndex, onHover }) {
  const canvasRef = useRef(null)
  const imgRef = useRef(null)
  const [imgLoaded, setImgLoaded] = useState(false)
  const [tooltip, setTooltip] = useState(null) // {x, y, det}

  // Load the processed PNG
  useEffect(() => {
    if (!jobId) return
    const img = new Image()
    img.src = `/api/image/${jobId}`
    img.onload = () => {
      imgRef.current = img
      setImgLoaded(true)
    }
  }, [jobId])

  // Redraw whenever image or detections change
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !imgRef.current) return

    const img = imgRef.current
    canvas.width = canvas.clientWidth
    canvas.height = Math.round((canvas.clientWidth / img.naturalWidth) * img.naturalHeight)

    const ctx = canvas.getContext('2d')
    const scaleX = canvas.width / img.naturalWidth
    const scaleY = canvas.height / img.naturalHeight

    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height)

    detections.forEach((det, i) => {
      if (det.obb) {
        drawOBB(ctx, det.obb, det.confidence ?? 1, i === highlightIndex, scaleX, scaleY)
      }
    })
  }, [imgLoaded, detections, highlightIndex])

  // Hit-test OBBs on mouse move to show tooltip
  const onMouseMove = (e) => {
    if (!canvasRef.current || !imgRef.current) return
    const rect = canvasRef.current.getBoundingClientRect()
    const mx = e.clientX - rect.left
    const my = e.clientY - rect.top
    const scaleX = canvasRef.current.width / imgRef.current.naturalWidth
    const scaleY = canvasRef.current.height / imgRef.current.naturalHeight

    let hit = null
    detections.forEach((det, i) => {
      const obb = det.obb
      if (!obb) return
      const dx = mx - obb.cx * scaleX
      const dy = my - obb.cy * scaleY
      const rad = -(obb.angle_deg * Math.PI) / 180
      const lx = dx * Math.cos(rad) - dy * Math.sin(rad)
      const ly = dx * Math.sin(rad) + dy * Math.cos(rad)
      if (Math.abs(lx) < (obb.w / 2) * scaleX && Math.abs(ly) < (obb.h / 2) * scaleY) {
        hit = { index: i, x: e.clientX, y: e.clientY, det }
      }
    })

    if (hit) {
      setTooltip(hit)
      onHover?.(hit.index)
    } else {
      setTooltip(null)
      onHover?.(null)
    }
  }

  if (!jobId) return null

  return (
    <div className="relative">
      {!imgLoaded && (
        <div className="flex items-center justify-center h-48 bg-slate-900 rounded-xl text-slate-400 text-sm">
          Loading image…
        </div>
      )}

      <canvas
        ref={canvasRef}
        className={`w-full rounded-xl ${imgLoaded ? 'block' : 'hidden'}`}
        onMouseMove={onMouseMove}
        onMouseLeave={() => { setTooltip(null); onHover?.(null) }}
      />

      {tooltip && (
        <div
          className="fixed z-50 bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-xs text-slate-200 pointer-events-none shadow-xl"
          style={{ left: tooltip.x + 14, top: tooltip.y - 10 }}
        >
          <p className="font-semibold text-cyan-300 mb-1">
            Detection {tooltip.index + 1}
          </p>
          <p>Confidence: {(tooltip.det.confidence * 100).toFixed(1)}%</p>
          <p>Length: {tooltip.det.streak_length_px?.toFixed(0)} px</p>
          {tooltip.det.ra_deg != null && (
            <p>RA / Dec: {tooltip.det.ra_deg.toFixed(4)}° / {tooltip.det.dec_deg.toFixed(4)}°</p>
          )}
          {tooltip.det.identifications?.[0] && (
            <p className="mt-1 text-yellow-300">
              {tooltip.det.identifications[0].satellite_name}
              {' '}({(tooltip.det.identifications[0].confidence * 100).toFixed(0)}%)
            </p>
          )}
        </div>
      )}
    </div>
  )
}
