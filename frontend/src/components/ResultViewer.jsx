import { useEffect, useRef, useState } from 'react'

const OBB_COLOUR = '#00DCFF'
const HIGHLIGHT_COLOUR = '#FF6B35'

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

function streakEndpoints(obb, scaleX, scaleY) {
  const { cx, cy, w, h, angle_deg } = obb
  const rad = (angle_deg * Math.PI) / 180
  // Use the longer OBB dimension as the streak half-length.  For correctly
  // processed detections w is the along-streak extent; for legacy DB rows
  // where w/h may be the raw bbox extents, max() picks the right one.
  const half = Math.max(w, h) / 2
  return {
    p1: { x: (cx - half * Math.cos(rad)) * scaleX, y: (cy - half * Math.sin(rad)) * scaleY },
    p2: { x: (cx + half * Math.cos(rad)) * scaleX, y: (cy + half * Math.sin(rad)) * scaleY },
  }
}

// ---------------------------------------------------------------------------
// Drawing primitives
// ---------------------------------------------------------------------------

function drawOBB(ctx, obb, colour, alpha, scaleX, scaleY, lineWidth) {
  const { cx, cy, w, h, angle_deg } = obb
  const rad = (angle_deg * Math.PI) / 180
  ctx.save()
  ctx.globalAlpha = alpha
  ctx.strokeStyle = colour
  ctx.lineWidth = lineWidth
  ctx.translate(cx * scaleX, cy * scaleY)
  ctx.rotate(rad)
  ctx.strokeRect((-w / 2) * scaleX, (-h / 2) * scaleY, w * scaleX, h * scaleY)
  ctx.restore()
}

function drawCenterline(ctx, p1, p2, colour, alpha, lineWidth) {
  ctx.save()
  ctx.globalAlpha = alpha
  ctx.strokeStyle = colour
  ctx.lineWidth = lineWidth
  ctx.setLineDash([6, 3])
  ctx.beginPath()
  ctx.moveTo(p1.x, p1.y)
  ctx.lineTo(p2.x, p2.y)
  ctx.stroke()
  ctx.setLineDash([])
  ctx.restore()
}

function drawEndpoint(ctx, p, label, colour, alpha, radius) {
  ctx.save()
  ctx.globalAlpha = alpha
  // Outer filled circle
  ctx.fillStyle = colour
  ctx.beginPath()
  ctx.arc(p.x, p.y, radius, 0, Math.PI * 2)
  ctx.fill()
  // Inner dark ring for contrast
  ctx.fillStyle = '#000'
  ctx.globalAlpha = alpha * 0.55
  ctx.beginPath()
  ctx.arc(p.x, p.y, radius * 0.42, 0, Math.PI * 2)
  ctx.fill()
  // Label ("1" / "2") just above the dot
  ctx.globalAlpha = alpha
  ctx.fillStyle = colour
  ctx.font = `bold ${Math.round(radius * 1.6)}px system-ui, sans-serif`
  ctx.textAlign = 'center'
  ctx.textBaseline = 'bottom'
  ctx.fillText(label, p.x, p.y - radius - 1)
  ctx.restore()
}

function drawAngleIndicator(ctx, obb, colour, alpha, scaleX, scaleY) {
  const { cx, cy, w, h, angle_deg } = obb
  const rad = (angle_deg * Math.PI) / 180
  const cxS = cx * scaleX
  const cyS = cy * scaleY

  // Arc radius: proportional to streak length, clamped
  const arcR = Math.max(16, Math.min(28, (Math.max(w, h) / 2) * scaleX * 0.28))

  ctx.save()
  ctx.globalAlpha = alpha * 0.75
  ctx.strokeStyle = colour
  ctx.fillStyle = colour
  ctx.lineWidth = 1

  // Horizontal reference tick from centre
  ctx.beginPath()
  ctx.moveTo(cxS, cyS)
  ctx.lineTo(cxS + arcR + 4, cyS)
  ctx.stroke()

  // Arc from 0 → angle_deg (counterclockwise if negative)
  ctx.beginPath()
  if (rad >= 0) {
    ctx.arc(cxS, cyS, arcR, 0, rad)
  } else {
    ctx.arc(cxS, cyS, arcR, rad, 0)
  }
  ctx.stroke()

  // Angle text at midpoint of arc
  const midAngle = rad / 2
  const labelR = arcR + 11
  ctx.globalAlpha = alpha
  ctx.font = 'bold 10px system-ui, sans-serif'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText(
    `${angle_deg.toFixed(1)}°`,
    cxS + labelR * Math.cos(midAngle),
    cyS + labelR * Math.sin(midAngle),
  )

  ctx.restore()
}

function drawLabel(ctx, obb, index, colour, alpha, scaleX, scaleY) {
  const { cx, cy, h } = obb
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
  ctx.globalAlpha = alpha

  ctx.fillStyle = colour
  roundRect(ctx, x - bw / 2, y - bh, bw, bh, 3)
  ctx.fill()

  ctx.fillStyle = colour === HIGHLIGHT_COLOUR ? '#fff' : '#000'
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

// ---------------------------------------------------------------------------
// Composite draw: one detection
// ---------------------------------------------------------------------------

function drawDetection(ctx, det, index, highlighted, scaleX, scaleY) {
  const { obb, confidence: conf = 1 } = det
  if (!obb || [obb.cx, obb.cy, obb.w, obb.h, obb.angle_deg].some((v) => v == null)) return

  const colour = highlighted ? HIGHLIGHT_COLOUR : OBB_COLOUR
  const alpha = highlighted ? 1.0 : 0.4 + conf * 0.6
  const endpointR = highlighted ? 5.5 : 4
  const lineWidth = highlighted ? 2.5 : 1.5

  const { p1, p2 } = streakEndpoints(obb, scaleX, scaleY)

  drawOBB(ctx, obb, colour, alpha * 0.55, scaleX, scaleY, highlighted ? 1.5 : 1)
  drawCenterline(ctx, p1, p2, colour, alpha, lineWidth)
  drawEndpoint(ctx, p1, '1', colour, alpha, endpointR)
  drawEndpoint(ctx, p2, '2', colour, alpha, endpointR)
  drawAngleIndicator(ctx, obb, colour, alpha, scaleX, scaleY)
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

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
  const [tooltip, setTooltip] = useState(null)

  useEffect(() => {
    if (!jobId) return
    const img = new Image()
    img.src = `/api/preview/${jobId}`
    img.onload = () => {
      imgRef.current = img
      setImgLoaded(true)
    }
  }, [jobId])

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

    // Non-highlighted detections first, then highlighted on top
    detections.forEach((det, i) => {
      if (i !== highlightIndex) drawDetection(ctx, det, i, false, scaleX, scaleY)
    })
    detections.forEach((det, i) => {
      if (i === highlightIndex) drawDetection(ctx, det, i, true, scaleX, scaleY)
    })

    // Labels always on top of everything
    detections.forEach((det, i) => {
      if (det.obb) {
        const colour = i === highlightIndex ? HIGHLIGHT_COLOUR : OBB_COLOUR
        const alpha = i === highlightIndex ? 1.0 : 0.4 + (det.confidence ?? 1) * 0.6
        drawLabel(ctx, det.obb, i, colour, alpha, scaleX, scaleY)
      }
    })
  }, [imgLoaded, detections, highlightIndex])

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
          Loading image…
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
          <p className="font-semibold text-cyan-300 mb-1.5">Detection {tooltip.index + 1}</p>
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
              <p>Angle: <span className="font-mono">{tooltip.det.obb.angle_deg.toFixed(1)}° from horizontal</span></p>
            )}
            {tooltip.det.ra_tip1_deg != null && (
              <p>
                Soln 1 RA / Dec:{' '}
                <span className="font-mono">
                  {tooltip.det.ra_tip1_deg.toFixed(4)}° / {tooltip.det.dec_tip1_deg.toFixed(4)}°
                </span>
              </p>
            )}
            {tooltip.det.ra_tip2_deg != null && (
              <p>
                Soln 2 RA / Dec:{' '}
                <span className="font-mono">
                  {tooltip.det.ra_tip2_deg.toFixed(4)}° / {tooltip.det.dec_tip2_deg.toFixed(4)}°
                </span>
              </p>
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

      {/* Legend */}
      {imgLoaded && detections.length > 0 && (
        <div className="absolute bottom-3 right-3 bg-slate-900/80 backdrop-blur-sm border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-400 flex flex-col gap-1.5">
          <div className="flex items-center gap-2">
            <span className="inline-block w-6 border-t-2 border-dashed border-cyan-400" />
            <span>Streak axis</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center justify-center w-3 h-3 rounded-full bg-cyan-400 text-[7px] font-bold text-black">1</span>
            <span>Soln 1 / 2 positions</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="font-mono text-cyan-400">θ°</span>
            <span>Angle from horizontal</span>
          </div>
        </div>
      )}
    </div>
  )
}
