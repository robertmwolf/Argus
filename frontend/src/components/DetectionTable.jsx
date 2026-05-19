/**
 * DetectionTable — one row per streak, showing every detector that fired on it.
 *
 * Props:
 *   detections     — array of streak dicts from /api/result (one per streak_id)
 *   highlightIndex — index of the row to highlight (synced with canvas hover)
 *   onRowClick(i)  — called when a row is clicked
 *   photoTakenAt   — observation DATE-OBS fallback from /api/result
 */

import { useEffect, useState } from 'react'

const METHOD_CONFIG = {
  unified:                 { label: 'Confidence Score',                    cls: 'border-emerald-500/80 bg-emerald-900/50 text-emerald-300' },
  astride:                 { label: 'ASTRiDE',                             cls: 'border-amber-600/60 bg-amber-950/40 text-amber-300' },
  opencv:                  { label: 'OpenCV',                              cls: 'border-orange-600/60 bg-orange-950/40 text-orange-300' },
  dinov3_vitb:             { label: 'DINOv3 ViT-Base - SatStreaks+GTImages', cls: 'border-cyan-600/60 bg-cyan-950/40 text-cyan-300' },
  dinov3_vitl:             { label: 'DINOv3 ViT-Large - SatStreaks+GTImages', cls: 'border-cyan-600/60 bg-cyan-950/40 text-cyan-300' },
  dinov3_gt_dm_satstreaks: { label: 'DINOv3 - GT+DM+SatStreaks',           cls: 'border-teal-600/60 bg-teal-950/40 text-teal-300' },
  streakmind_yolo:         { label: 'YOLO-OBB - GTImages',                 cls: 'border-fuchsia-500/70 bg-fuchsia-950/40 text-fuchsia-300' },
  yolo:                    { label: 'YOLO-OBB - SatStreaks (dev)',          cls: 'border-violet-600/60 bg-violet-950/40 text-violet-300' },
  yolo_full:               { label: 'YOLO-OBB - SatStreaks',               cls: 'border-purple-600/60 bg-purple-950/40 text-purple-300' },
  tiny:                    { label: 'DINO Swin-Tiny - SatStreaks',          cls: 'border-cyan-600/60 bg-cyan-950/40 text-cyan-300' },
  large:                   { label: 'DINO Swin-Large - SatStreaks',         cls: 'border-cyan-600/60 bg-cyan-950/40 text-cyan-300' },
  classical:               { label: 'Classical',                           cls: 'border-amber-600/60 bg-amber-950/40 text-amber-300' },
  ml:                      { label: 'ML',                                  cls: 'border-cyan-600/60 bg-cyan-950/40 text-cyan-300' },
}

function MethodBadge({ method, confidence }) {
  const m = METHOD_CONFIG[method] ?? {
    label: method ?? 'Unknown',
    cls: 'border-slate-600/60 bg-slate-800/40 text-slate-400',
  }
  return (
    <div className="flex items-center gap-1.5">
      <span className={`inline-flex items-center rounded border px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide ${m.cls}`}>
        {m.label}
      </span>
      <span className={[
        'text-xs tabular-nums font-medium',
        confidence >= 0.9 ? 'text-green-400' :
        confidence >= 0.7 ? 'text-yellow-400' :
        'text-red-400',
      ].join(' ')}>
        {(confidence * 100).toFixed(1)}%
      </span>
    </div>
  )
}

function EyeIcon({ visible }) {
  return visible ? (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M2.036 12.322a1.012 1.012 0 0 1 0-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178Z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
    </svg>
  ) : (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M3.98 8.223A10.477 10.477 0 0 0 1.934 12C3.226 16.338 7.244 19.5 12 19.5c.993 0 1.953-.138 2.863-.395M6.228 6.228A10.451 10.451 0 0 1 12 4.5c4.756 0 8.773 3.162 10.065 7.498a10.522 10.522 0 0 1-4.293 5.774M6.228 6.228 3 3m3.228 3.228 3.65 3.65m7.894 7.894L21 21m-3.228-3.228-3.65-3.65m0 0a3 3 0 1 0-4.243-4.243m4.242 4.242L9.88 9.88" />
    </svg>
  )
}

function formatDateTime(value) {
  if (!value) return '—'
  const dt = new Date(value)
  if (Number.isNaN(dt.getTime())) return value
  return dt.toLocaleString(undefined, {
    year: '2-digit',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function formatTleCurrency(best) {
  if (!best) return '—'
  const age = best.tle_age_hours ?? best.epoch_drift_hours
  const epoch = formatDateTime(best.tle_epoch)
  if (age == null) return epoch
  const absAge = Math.abs(age)
  const unitValue = absAge >= 48 ? absAge / 24 : absAge
  const unit = absAge >= 48 ? 'd' : 'h'
  return `${epoch} (${unitValue.toFixed(absAge >= 48 ? 1 : 0)} ${unit})`
}

export default function DetectionTable({ detections, visibleSet, highlightIndex, onRowClick, onToggleStreak, photoTakenAt }) {
  const [headerPhotoDate, setHeaderPhotoDate] = useState(null)

  useEffect(() => {
    if (photoTakenAt) {
      setHeaderPhotoDate(null)
      return
    }

    const rows = Array.from(document.querySelectorAll('table tbody tr'))
    const dateObsRow = rows.find((row) => {
      const firstCell = row.children[0]?.textContent?.trim()
      return firstCell === 'DATE-OBS'
    })
    const dateObs = dateObsRow?.children[1]?.textContent?.trim()
    setHeaderPhotoDate(dateObs ? normaliseDateObs(dateObs) : null)
  }, [photoTakenAt, detections])

  if (!detections || detections.length === 0) {
    return (
      <div className="rounded-xl border border-slate-700 px-4 py-8 text-center">
        <p className="text-slate-500 text-sm">No streaks detected in this image.</p>
      </div>
    )
  }

  return (
    <div className="rounded-xl border border-slate-700 overflow-hidden">
      <div className="px-4 py-3 bg-slate-800/60 border-b border-slate-700 flex items-center gap-2">
        <svg className="w-4 h-4 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
            d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25H12" />
        </svg>
        <h3 className="text-sm font-semibold text-slate-300">Streak Detections</h3>
        <span className="text-xs text-slate-500">
          {detections.length} streak{detections.length !== 1 ? 's' : ''}
        </span>
        <span className="ml-auto text-xs text-slate-600">Click a row to highlight on image</span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm text-left">
          <thead className="bg-slate-800/40 text-slate-400 text-xs uppercase tracking-wider">
            <tr>
              {[
                { label: '' },
                { label: '#' },
                { label: 'Confidence Score' },
                { label: 'Length (px)' },
                { label: 'Angle (°)' },
                { label: 'Sky Position' },
                { label: 'Photo Date' },
                { label: 'Best Match', divider: true },
                { label: 'TLE Data', sub: 'epoch age', divider: true },
                { label: 'ID Confidence', sub: 'position × length', divider: true },
              ].map(({ label, sub, divider }) => (
                <th
                  key={label || '_toggle'}
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
            {detections.map((det, idx) => {
              const best = det.identifications?.[0]
              const isHighlighted = idx === highlightIndex
              const isVisible = !visibleSet || visibleSet.has(idx)
              const angleDeg = det.obb?.angle_deg
              const sources = det.sources ?? [{ method: det.method, confidence: det.confidence }]
              const obbMax = det.obb ? Math.max(det.obb.w ?? 0, det.obb.h ?? 0) : 0
              const len = (det.streak_length_px != null && det.streak_length_px > obbMax * 0.1)
                ? det.streak_length_px : obbMax

              return (
                <tr
                  key={idx}
                  onClick={() => onRowClick?.(idx === highlightIndex ? null : idx)}
                  className={[
                    'cursor-pointer transition-colors border-t border-slate-800',
                    !isVisible
                      ? 'opacity-40'
                      : '',
                    isHighlighted
                      ? 'bg-orange-950/40 text-orange-200'
                      : 'hover:bg-slate-800/50 text-slate-300',
                  ].join(' ')}
                >
                  {/* Eye toggle */}
                  <td className="px-2 py-2.5">
                    <button
                      onClick={e => { e.stopPropagation(); onToggleStreak?.(idx) }}
                      title={isVisible ? 'Hide streak' : 'Show streak'}
                      className={`p-1 rounded transition-colors ${
                        isVisible
                          ? 'text-slate-400 hover:text-white'
                          : 'text-slate-700 hover:text-slate-400'
                      }`}
                    >
                      <EyeIcon visible={isVisible} />
                    </button>
                  </td>

                  {/* Index */}
                  <td className="px-4 py-2.5">
                    <span className={`inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold ${
                      isHighlighted ? 'bg-orange-500/20 text-orange-300' : 'bg-slate-800 text-slate-400'
                    }`}>
                      {idx + 1}
                    </span>
                  </td>

                  {/* Multi-method sources — Unified always first, then individual methods */}
                  <td className="px-4 py-2.5">
                    <div className="flex flex-col gap-1">
                      {sources.map((src, si) => (
                        <div key={si}>
                          <MethodBadge method={src.method} confidence={src.confidence} />
                          {si === 0 && sources.length > 1 && (
                            <div className="mt-1 mb-0.5 border-t border-slate-700/60" />
                          )}
                        </div>
                      ))}
                    </div>
                  </td>

                  {/* Length */}
                  <td className="px-4 py-2.5 font-mono text-slate-300">
                    {len > 0 ? len.toFixed(0) : '—'}
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

                  {/* Photo date */}
                  <td className="px-4 py-2.5 font-mono text-xs text-slate-400 whitespace-nowrap">
                    {formatDateTime(best?.photo_taken_at ?? det.photo_taken_at ?? photoTakenAt ?? headerPhotoDate)}
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

                  {/* TLE data currency */}
                  <td className="px-4 py-2.5 border-l border-slate-600/60">
                    <div className="flex flex-col gap-0.5">
                      <span className="font-mono text-xs text-slate-300 whitespace-nowrap">
                        {formatTleCurrency(best)}
                      </span>
                      {best?.tle_search_mode && best.tle_search_mode !== 'normal' && (
                        <span className="text-[10px] uppercase tracking-wide text-amber-400">
                          {best.tle_search_mode.replaceAll('_', ' ')}
                        </span>
                      )}
                    </div>
                  </td>

                  {/* ID confidence */}
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

function normaliseDateObs(value) {
  if (!value) return null
  const raw = String(value).trim().replace(' ', 'T')
  if (/[zZ]$|[+-]\d{2}:?\d{2}$/.test(raw)) return raw
  return `${raw}Z`
}
