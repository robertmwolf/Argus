/**
 * IdentificationPanel — shows satellite identification candidates for the
 * currently selected detection.
 *
 * Props:
 *   detections  — array of detection dicts from /api/result
 *   activeIndex — index of the selected detection (or null)
 *
 * Candidate rows document the confidence calculation as rotation fit ×
 * lateral fit × TLE-age fit, with signed residuals shown in arcseconds.
 */
export default function IdentificationPanel({ detections, activeIndex }) {
  if (!detections || detections.length === 0) return null

  const allHaveNoIds = detections.every((d) => !d.identifications?.length)
  if (allHaveNoIds) return null

  const det = activeIndex != null ? detections[activeIndex] : null
  const ids = det?.identifications ?? []

  return (
    <div className="rounded-xl border border-slate-700 overflow-hidden">
      {/* Panel header */}
      <div className="px-4 py-3 bg-slate-800/60 border-b border-slate-700 flex items-center gap-3">
        <svg className="w-4 h-4 text-yellow-400 flex-none" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
            d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
        </svg>
        <h3 className="text-sm font-semibold text-slate-300">Satellite Identification Candidates</h3>
        {det && (
          <span className="ml-auto text-xs text-slate-500">
            Detection {activeIndex + 1}
          </span>
        )}
      </div>

      {/* Empty-state when no detection is selected */}
      {!det && (
        <div className="px-4 py-8 text-center text-slate-500 text-sm">
          Click a row in the detections table to view candidate satellite matches.
        </div>
      )}

      {/* Selected detection with no candidates */}
      {det && ids.length === 0 && (
        <div className="px-4 py-8 text-center">
          <p className="text-slate-500 text-sm">No satellite candidates matched for detection {activeIndex + 1}.</p>
          <p className="text-slate-600 text-xs mt-1">The streak may be uncatalogued or TLE data was unavailable.</p>
        </div>
      )}

      {/* Candidate list */}
      {det && ids.length > 0 && (
        <div className="divide-y divide-slate-800">
          {ids.map((id, i) => (
            <CandidateRow key={i} id={id} rank={i} />
          ))}
        </div>
      )}
    </div>
  )
}

function CandidateRow({ id, rank }) {
  const isBest = rank === 0
  const conf = id.confidence ?? 0
  const confPct = Math.round(conf * 100)

  const confColour =
    conf >= 0.8 ? 'text-green-400' :
    conf >= 0.5 ? 'text-yellow-400' :
    'text-red-400'

  const barColour =
    conf >= 0.8 ? 'bg-green-500' :
    conf >= 0.5 ? 'bg-yellow-500' :
    'bg-red-500'

  const photoDate = formatDateTime(id.photo_taken_at)
  const tleEpoch = formatDateTime(id.tle_epoch)
  const ageHours = id.tle_age_hours ?? id.epoch_drift_hours
  const ageLabel = ageHours == null
    ? null
    : `${Math.abs(ageHours) >= 48 ? (Math.abs(ageHours) / 24).toFixed(1) + ' d' : Math.abs(ageHours).toFixed(0) + ' h'} TLE age`

  const factors = [
    {
      label: 'Rotation fit',
      score: id.rotation_score,
      detail: formatSignedArcsec(id.atrk_arcsec),
      title: 'Along-track difference: how far ahead or behind the TLE prediction the streak appears.',
    },
    {
      label: 'Lateral fit',
      score: id.lateral_score,
      detail: formatSignedArcsec(id.xtrk_arcsec),
      title: 'Cross-track difference: sideways displacement from the predicted orbital path.',
    },
    {
      label: 'TLE age fit',
      score: id.epoch_penalty,
      detail: ageLabel,
      title: 'Confidence retained after accounting for the age of the orbital elements.',
    },
  ]

  return (
    <div className={`px-4 py-3 flex items-center gap-4 ${isBest ? 'bg-yellow-950/20' : 'hover:bg-slate-800/30'} transition-colors`}>
      {/* Rank badge */}
      <div className={`flex-none w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold border ${
        isBest
          ? 'bg-yellow-900/40 border-yellow-600/50 text-yellow-300'
          : 'bg-slate-800 border-slate-600 text-slate-400'
      }`}>
        {id.rank ?? rank + 1}
      </div>

      {/* Name + metadata */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className={`font-semibold text-sm truncate ${isBest ? 'text-yellow-200' : 'text-slate-300'}`}>
            {id.satellite_name ?? `NORAD ${id.norad_id}`}
          </span>
          {isBest && (
            <span className="text-xs bg-yellow-900/40 text-yellow-400 border border-yellow-700/40 rounded-full px-2 py-0.5 whitespace-nowrap">
              Best match
            </span>
          )}
        </div>
        <div className="flex items-center gap-4 mt-0.5 text-xs text-slate-500 flex-wrap">
          {id.norad_id != null && (
            <span>NORAD {id.norad_id}</span>
          )}
          {id.separation_deg != null && (
            <span title="Angular separation between detection centroid and predicted TLE position">
              Δ {(id.separation_deg * 3600).toFixed(1)}″ sep.
            </span>
          )}
          {photoDate !== '—' && <span>Photo {photoDate}</span>}
          {tleEpoch !== '—' && <span>TLE {tleEpoch}</span>}
          {ageLabel && <span>{ageLabel}</span>}
        </div>
        <div className="mt-2 grid grid-cols-1 sm:grid-cols-3 gap-1.5 max-w-2xl">
          {factors.map((factor) => (
            <ConfidenceFactor key={factor.label} {...factor} />
          ))}
        </div>
        {id.confidence_method === 'rotation_x_lateral_x_tle_age' && (
          <div className="mt-1.5 text-[10px] text-slate-600">
            Match confidence = rotation fit × lateral fit × TLE age fit
          </div>
        )}
        {id.confidence_method === 'position_x_tle_age' && (
          <div className="mt-1.5 text-[10px] text-amber-500/80">
            Clipped streak: rotation and lateral factors unavailable; visible-position fit × TLE age used.
          </div>
        )}
      </div>

      {/* Confidence */}
      <div className="flex-none text-right">
        <div className={`text-sm font-semibold tabular-nums ${confColour}`}>
          {id.confidence != null ? `${confPct}%` : '—'}
        </div>
        <div className="mt-1.5 w-20 h-1.5 bg-slate-700 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all ${barColour}`}
            style={{ width: `${confPct}%` }}
          />
        </div>
      </div>
    </div>
  )
}

function ConfidenceFactor({ label, score, detail, title }) {
  const colour = score == null
    ? 'text-slate-600'
    : score >= 0.8 ? 'text-green-400'
    : score >= 0.5 ? 'text-yellow-400'
    : 'text-red-400'
  return (
    <div title={title} className="rounded border border-slate-700/70 bg-slate-900/30 px-2 py-1">
      <div className="flex items-center justify-between gap-2 text-[10px]">
        <span className="text-slate-500">{label}</span>
        <span className={`font-mono font-medium ${colour}`}>
          {score == null ? '—' : `${(score * 100).toFixed(0)}%`}
        </span>
      </div>
      {detail && <div className="mt-0.5 text-[10px] font-mono text-slate-500 truncate">{detail}</div>}
    </div>
  )
}

function formatSignedArcsec(value) {
  if (value == null) return null
  return `${value >= 0 ? '+' : ''}${value.toFixed(1)}″`
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
