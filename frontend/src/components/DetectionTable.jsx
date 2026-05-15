/**
 * DetectionTable — tabular view of pipeline detections.
 *
 * Props:
 *   detections     — array of detection dicts from /api/result
 *   highlightIndex — index of the row to highlight (synced with canvas hover)
 *   onRowClick(i)  — called when a row is clicked
 */
const TOP_PER_METHOD = 3

const METHOD_CONFIG = {
  astride:      { label: 'ASTRiDE',       cls: 'border-amber-600/60 bg-amber-950/40 text-amber-300' },
  opencv:       { label: 'OpenCV',        cls: 'border-orange-600/60 bg-orange-950/40 text-orange-300' },
  dinov3_vitb:  { label: 'DINOv3 ViT-B', cls: 'border-cyan-600/60 bg-cyan-950/40 text-cyan-300' },
  dinov3_vitl:  { label: 'DINOv3 ViT-L', cls: 'border-cyan-600/60 bg-cyan-950/40 text-cyan-300' },
  tiny:         { label: 'DINO Swin-T',  cls: 'border-cyan-600/60 bg-cyan-950/40 text-cyan-300' },
  large:        { label: 'DINO Swin-L',  cls: 'border-cyan-600/60 bg-cyan-950/40 text-cyan-300' },
  // legacy fallbacks for detections stored before this change
  classical:    { label: 'Classical',    cls: 'border-amber-600/60 bg-amber-950/40 text-amber-300' },
  ml:           { label: 'ML',           cls: 'border-cyan-600/60 bg-cyan-950/40 text-cyan-300' },
}

export default function DetectionTable({ detections, totalDetections, highlightIndex, onRowClick }) {
  if (!detections || detections.length === 0) {
    return (
      <div className="rounded-xl border border-slate-700 px-4 py-8 text-center">
        <p className="text-slate-500 text-sm">No streaks detected in this image.</p>
      </div>
    )
  }

  // detections arrives pre-filtered and pre-sorted from App.jsx (top N per
  // method, length desc). We only need to group by method for visual separation.
  const methodGroups = new Map()
  for (const det of detections) {
    const key = det.method ?? 'unknown'
    if (!methodGroups.has(key)) methodGroups.set(key, [])
    methodGroups.get(key).push(det)
  }

  // Flat list preserving the original order, annotated with rank-within-method
  const flatRows = []
  for (const [methodKey, dets] of methodGroups) {
    dets.forEach((det, rank) => flatRows.push({ det, methodKey, rankInMethod: rank }))
  }

  const totalMethods = methodGroups.size

  return (
    <div className="rounded-xl border border-slate-700 overflow-hidden">
      <div className="px-4 py-3 bg-slate-800/60 border-b border-slate-700 flex items-center gap-2">
        <svg className="w-4 h-4 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
            d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25H12" />
        </svg>
        <h3 className="text-sm font-semibold text-slate-300">Streak Detections</h3>
        <span className="text-xs text-slate-500">
          top {TOP_PER_METHOD} per method · {totalMethods} method{totalMethods !== 1 ? 's' : ''} · {detections.length} shown of {totalDetections ?? detections.length}
        </span>
        <span className="ml-auto text-xs text-slate-600">Click a row to highlight on image</span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm text-left">
          <thead className="bg-slate-800/40 text-slate-400 text-xs uppercase tracking-wider">
            <tr>
              {[
                { label: '#' },
                { label: 'Method' },
                { label: 'Det. Confidence', sub: 'model / aspect score' },
                { label: 'Length (px)' },
                { label: 'Angle (°)' },
                { label: 'Sky Position' },
                { label: 'Best Match', divider: true },
                { label: 'ID Confidence', sub: 'position × length', divider: true },
              ].map(({ label, sub, divider }) => (
                <th
                  key={label}
                  className={[
                    'px-4 py-2.5 whitespace-nowrap font-medium',
                    divider ? 'border-l border-slate-600/60' : '',
                  ].join(' ')}
                >
                  <div className="leading-none">{label}</div>
                  {sub && <div className="mt-0.5 text-[10px] normal-case tracking-normal text-slate-500 font-normal">{sub}</div>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {flatRows.map(({ det, methodKey, rankInMethod }, flatIdx) => {
              const best = det.identifications?.[0]
              const isHighlighted = flatIdx === highlightIndex
              const angleDeg = det.obb?.angle_deg
              const isFirstInMethod = rankInMethod === 0
              const isNewMethodGroup = isFirstInMethod && flatIdx > 0

              return (
                <tr
                  key={flatIdx}
                  onClick={() => onRowClick?.(flatIdx === highlightIndex ? null : flatIdx)}
                  className={[
                    'cursor-pointer transition-colors',
                    isNewMethodGroup ? 'border-t-2 border-slate-600' : 'border-t border-slate-800',
                    isHighlighted
                      ? 'bg-orange-950/40 text-orange-200'
                      : 'hover:bg-slate-800/50 text-slate-300',
                  ].join(' ')}
                >
                  {/* Rank within method — shown as 1/2/3 */}
                  <td className="px-4 py-2.5">
                    <span className={`inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold ${
                      isHighlighted ? 'bg-orange-500/20 text-orange-300' : 'bg-slate-800 text-slate-400'
                    }`}>
                      {flatIdx + 1}
                    </span>
                  </td>

                  {/* Detection method */}
                  <td className="px-4 py-2.5">
                    {(() => {
                      const m = METHOD_CONFIG[det.method] ?? { label: det.method ?? 'Unknown', cls: 'border-slate-600/60 bg-slate-800/40 text-slate-400' }
                      return (
                        <span className={`inline-flex items-center rounded border px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide ${m.cls}`}>
                          {m.label}
                        </span>
                      )
                    })()}
                  </td>

                  {/* Confidence */}
                  <td className="px-4 py-2.5">
                    <div className="flex items-center gap-2">
                      <span className={[
                        'font-semibold tabular-nums',
                        det.confidence >= 0.9 ? 'text-green-400' :
                        det.confidence >= 0.7 ? 'text-yellow-400' :
                        'text-red-400',
                      ].join(' ')}>
                        {(det.confidence * 100).toFixed(1)}%
                      </span>
                      <div className="w-12 h-1 bg-slate-700 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${
                            det.confidence >= 0.9 ? 'bg-green-500' :
                            det.confidence >= 0.7 ? 'bg-yellow-500' :
                            'bg-red-500'
                          }`}
                          style={{ width: `${det.confidence * 100}%` }}
                        />
                      </div>
                    </div>
                  </td>

                  {/* Length */}
                  <td className="px-4 py-2.5 font-mono text-slate-300">
                    {(() => {
                      const obbMax = det.obb ? Math.max(det.obb.w ?? 0, det.obb.h ?? 0) : 0
                      const len = (det.streak_length_px != null && det.streak_length_px > obbMax * 0.1)
                        ? det.streak_length_px
                        : obbMax
                      return len > 0 ? len.toFixed(0) : '—'
                    })()}
                  </td>

                  {/* Angle */}
                  <td className="px-4 py-2.5 font-mono text-slate-300">
                    {angleDeg != null ? angleDeg.toFixed(1) : '—'}
                  </td>

                  {/* Sky Position */}
                  <td className="px-4 py-2.5">
                    {det.ra_tip1_deg != null ? (
                      <div className="flex flex-col gap-0.5">
                        <div className="flex items-baseline gap-1.5">
                          <span className="inline-flex items-center justify-center w-3.5 h-3.5 rounded-full bg-cyan-700 text-cyan-200 text-[8px] font-bold shrink-0">1</span>
                          <span className="font-mono text-xs text-slate-300">
                            {det.ra_tip1_deg.toFixed(4)}°&nbsp;/&nbsp;{det.dec_tip1_deg.toFixed(4)}°
                          </span>
                        </div>
                        {det.ra_tip2_deg != null && (
                          <div className="flex items-baseline gap-1.5">
                            <span className="inline-flex items-center justify-center w-3.5 h-3.5 rounded-full bg-slate-600 text-slate-200 text-[8px] font-bold shrink-0">2</span>
                            <span className="font-mono text-xs text-slate-400">
                              {det.ra_tip2_deg.toFixed(4)}°&nbsp;/&nbsp;{det.dec_tip2_deg.toFixed(4)}°
                            </span>
                          </div>
                        )}
                      </div>
                    ) : (
                      <span className="text-slate-600">—</span>
                    )}
                  </td>

                  {/* Best match name */}
                  <td className="px-4 py-2.5 border-l border-slate-600/60">
                    {best ? (
                      <span className="text-yellow-300 font-medium">
                        {best.satellite_name ?? `NORAD ${best.norad_id}`}
                      </span>
                    ) : (
                      <span className="text-slate-600">Unidentified</span>
                    )}
                  </td>

                  {/* ID confidence — position × length Gaussian score from crossid.py */}
                  <td className="px-4 py-2.5 border-l border-slate-600/60">
                    {best?.confidence != null ? (
                      <span className={
                        best.confidence >= 0.8 ? 'text-green-400' :
                        best.confidence >= 0.5 ? 'text-yellow-400' :
                        'text-red-400'
                      }>
                        {(best.confidence * 100).toFixed(0)}%
                      </span>
                    ) : (
                      <span className="text-slate-600">—</span>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
