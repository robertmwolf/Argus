import { useEffect, useRef, useState } from 'react'

const OBB_COLOUR = '#00DCFF'
const HIGHLIGHT_COLOUR = '#FF6B35'
const LABEL_BG = 'rgba(0,0,0,0.65)'

/**
 * Draw one oriented bounding box on the canvas context.
 */
function drawOBB(ctx, obb, conf, highlighted, scaleX, scaleY) {
  const { cx, cy, w, h, angle_deg } = obb
  if ([cx, cy, w, h, angle_deg].some((v) => v == null)) return

  const rad = (angle_deg * Math.PI) / 180
  ctx.save()
  ctx.translate(cx * scaleX, cy * scaleY)
  ctx.rotate(rad)

  ctx.strokeStyle = highlighted ? HIGHLIGHT_COLOUR : OBB_COLOUR
  ctx.lineWidth = highlighted ? 3 : 1.5
  ctx.globalAlpha = highlighted ? 1.0 : 0.35 + conf * 0.65
  ctx.strokeRect((-w / 2) * scaleX, (-h / 2) * scaleY, w * scaleX, h * scaleY)

  ctx.restore()
}

/**
 * Draw a numbered label badge above the OBB centre.
 */
function drawLabel(ctx, obb, index, highlighted, scaleX, scaleY) {
  const { cx, cy, h } = obb
  if ([cx, cy, h].some((v) => v == null)) return

  const x = cx * scaleX
  const y = (cy - h / 2) * scaleY - 6

  const label = String(index + 1)
  const fontSize = 11
  ctx.font = `bold ${fontSize}px system-ui, sans-serif`
  const textW = ctx.measureText(label).width
  const pad = 4
  const bw = textW + pad * 2
  const bh = fontSize + pad * 2

  ctx.save()
  ctx.globalAlpha = highlighted ? 1.0 : 0.85

  // Badge background
  ctx.fillStyle = highlighted ? HIGHLIGHT_COLOUR : OBB_COLOUR
  roundRect(ctx, x - bw / 2, y - bh, bw, bh, 3)
  ctx.fill()

  // Badge text
  ctx.fillStyle = highlighted ? '#fff' : '#000'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText(label, x, y - bh / 2)

  ctx.restore()
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath()
  ctx.moveTo(x + r, y)
  ctx.lineTo(x + w - r, y)
  ctx.quadraticCurveTo(x + w, y, x + w, y + r)
  ctx.lineTo(x + w, y + h - r)
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h)
  ctx.lineTo(x + r, y + h)
  ctx.quadraticCurveTo(x, y + h, x, y + h - r)
  ctx.lineTo(x, y + r)
  ctx.quadraticCurveTo(x, y, x + r, y)
  ctx.closePath()
}

/**
 * ResultViewer — canvas overlay of detection OBBs on the result image.
 *
 * Props:
 *   jobId            — UUID of the completed job
 *   detections       — array of detection dicts from /api/result
 *   highlightIndex   — index of the detection to highlight (or null)
 *   onHover(index)   — called when mouse enters an OBB (null = leave)
 */
export default function ResultViewer({ jobId, detections, highlightIndex, onHover }) {
  const canvasRef = useRef(null)
  const imgRef = useRef(null)
  const [imgLoaded, setImgLoaded] = useState(false)
  const [tooltip, setTooltip] = useState(null) // {x, y, det, index}

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

  // Redraw when image, detections, or highlight changes
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

    // Draw all OBBs first, then labels on top
    detections.forEach((det, i) => {
      if (det.obb) drawOBB(ctx, det.obb, det.confidence ?? 1, i === highlightIndex, scaleX, scaleY)
    })
    detections.forEach((det, i) => {
      if (det.obb) drawLabel(ctx, det.obb, i, i === highlightIndex, scaleX, scaleY)
    })
  }, [imgLoaded, detections, highlightIndex])

  // OBB hit-test on mouse move
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
        <div className="flex items-center justify-center h-48 bg-slate-900 rounded-xl text-slate-500 text-sm gap-2">
          <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
          </svg>
          Loading result image…
        </div>
      )}

      <canvas
        ref={canvasRef}
        className={`w-full rounded-xl ${imgLoaded ? 'block' : 'hidden'}`}
        onMouseMove={onMouseMove}
        onMouseLeave={() => { setTooltip(null); onHover?.(null) }}
      />

      {/* Hover tooltip */}
      {tooltip && (
        <div
          className="fixed z-50 bg-slate-900 border border-slate-600 rounded-xl px-3 py-2.5 text-xs text-slate-200 pointer-events-none shadow-2xl"
          style={{ left: tooltip.x + 16, top: tooltip.y - 12 }}
        >
          <p className="font-semibold text-cyan-300 mb-1.5">
            Detection {tooltip.index + 1}
          </p>
          <div className="flex flex-col gap-0.5 text-slate-300">
            <p>
              Confidence:{' '}
              <span className={
                tooltip.det.confidence >= 0.9 ? 'text-green-400 font-semibold' :
                tooltip.det.confidence >= 0.7 ? 'text-yellow-400 font-semibold' :
                'text-red-400 font-semibold'
              }>
                {(tooltip.det.confidence * 100).toFixed(1)}%
              </span>
            </p>
            {tooltip.det.streak_length_px != null && (
              <p>Length: <span className="font-mono">{tooltip.det.streak_length_px.toFixed(0)} px</span></p>
            )}
            {tooltip.det.obb?.angle_deg != null && (
              <p>Angle: <span className="font-mono">{tooltip.det.obb.angle_deg.toFixed(1)}°</span></p>
            )}
            {tooltip.det.ra_deg != null && (
              <p>RA / Dec: <span className="font-mono">{tooltip.det.ra_deg.toFixed(4)}° / {tooltip.det.dec_deg.toFixed(4)}°</span></p>
            )}
          </div>
          {tooltip.det.identifications?.[0] && (
            <div className="mt-2 pt-2 border-t border-slate-700">
              <p className="text-yellow-300 font-semibold">
                {tooltip.det.identifications[0].satellite_name ?? `NORAD ${tooltip.det.identifications[0].norad_id}`}
              </p>
              <p className="text-slate-400">
                Match: {(tooltip.det.identifications[0].confidence * 100).toFixed(0)}%
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
