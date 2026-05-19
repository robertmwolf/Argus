/**
 * FilterPanel — per-method confidence sliders + streak count indicator.
 *
 * Props:
 *   detections       — full array of streak dicts (unfiltered)
 *   methodThresholds — { [method]: 0-1 }
 *   onThresholdChange(method, value) — called when a slider moves
 *   visibleCount     — number of streaks currently passing all filters
 */

const METHOD_LABELS = {
  unified:                 'Unified Confidence',
  astride:                 'ASTRiDE',
  opencv:                  'OpenCV',
  dinov3_vitb:             'DINOv3 ViT-Base - SatStreaks+GTImages',
  dinov3_vitl:             'DINOv3 ViT-Large - SatStreaks+GTImages',
  dinov3_gt_dm_satstreaks: 'DINOv3 - GT+DM+SatStreaks',
  streakmind_yolo:         'YOLO-OBB - GTImages',
  yolo:                    'YOLO-OBB - SatStreaks (dev)',
  yolo_full:               'YOLO-OBB - SatStreaks',
  tiny:                    'DINO Swin-Tiny - SatStreaks',
  large:                   'DINO Swin-Large - SatStreaks',
  classical:               'Classical',
  ml:                      'ML',
}

const METHOD_ACCENT = {
  unified:                 'accent-emerald-400',
  astride:                 'accent-amber-400',
  opencv:                  'accent-orange-400',
  dinov3_vitb:             'accent-cyan-400',
  dinov3_vitl:             'accent-cyan-400',
  dinov3_gt_dm_satstreaks: 'accent-teal-400',
  streakmind_yolo:         'accent-fuchsia-400',
  yolo:                    'accent-violet-400',
  yolo_full:               'accent-purple-400',
  tiny:                    'accent-cyan-400',
  large:                   'accent-cyan-400',
  classical:               'accent-amber-400',
  ml:                      'accent-cyan-400',
}

export default function FilterPanel({
  detections,
  methodThresholds,
  onThresholdChange,
  visibleCount,
}) {
  if (!detections || detections.length === 0) return null

  // Collect unique methods across all detections
  const methods = [...new Set(
    detections.flatMap(d => (d.sources ?? [{ method: d.method }]).map(s => s.method).filter(Boolean))
  )].sort()

  const total = detections.length

  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900/60 px-4 py-3 flex flex-col gap-3">
      {/* Header row */}
      <div className="flex items-center gap-3">
        <svg className="w-4 h-4 text-slate-400 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
            d="M3 4.5h14.25M3 9h9.75M3 13.5h5.25m5.25-.75L17.25 9m0 0L21 12.75M17.25 9v12" />
        </svg>
        <span className="text-sm font-semibold text-slate-300">Filters</span>
        <span className="text-xs text-slate-500">
          Showing{' '}
          <span className={visibleCount < total ? 'text-cyan-400 font-semibold' : 'text-slate-400'}>
            {visibleCount}
          </span>
          {' '}of {total} streak{total !== 1 ? 's' : ''}
        </span>
      </div>

      {/* Per-method sliders */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-x-6 gap-y-2.5">
        {methods.map(method => {
          const pct = Math.round((methodThresholds[method] ?? 0) * 100)
          const label = METHOD_LABELS[method] ?? method
          const accent = METHOD_ACCENT[method] ?? 'accent-slate-400'
          return (
            <div key={method} className="flex flex-col gap-1">
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-400 font-medium">{label}</span>
                <span className="text-xs font-mono text-slate-400 tabular-nums w-8 text-right">
                  {pct}%
                </span>
              </div>
              <input
                type="range"
                min={0}
                max={100}
                value={pct}
                onChange={e => onThresholdChange(method, Number(e.target.value) / 100)}
                className={`w-full h-1.5 rounded-full cursor-pointer bg-slate-700 ${accent}`}
              />
            </div>
          )
        })}
      </div>
    </div>
  )
}
