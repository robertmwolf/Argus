/**
 * DetectionTable — tabular view of pipeline detections.
 *
 * Props:
 *   detections     — array of detection dicts from /api/result
 *   highlightIndex — index of the row to highlight (synced with canvas hover)
 *   onRowClick(i)  — called when a row is clicked
 */
export default function DetectionTable({ detections, highlightIndex, onRowClick }) {
  if (!detections || detections.length === 0) {
    return (
      <p className="text-slate-500 text-sm text-center py-6">No detections found.</p>
    )
  }

  return (
    <div className="overflow-x-auto rounded-xl border border-slate-700">
      <table className="w-full text-sm text-left">
        <thead className="bg-slate-800 text-slate-400 text-xs uppercase tracking-wider">
          <tr>
            {['#', 'Conf', 'Length (px)', 'RA (°)', 'Dec (°)', 'Best ID', 'ID Conf'].map((h) => (
              <th key={h} className="px-4 py-3 whitespace-nowrap">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {detections.map((det, i) => {
            const best = det.identifications?.[0]
            const isHighlighted = i === highlightIndex
            return (
              <tr
                key={i}
                onClick={() => onRowClick?.(i)}
                className={[
                  'border-t border-slate-800 cursor-pointer transition-colors',
                  isHighlighted
                    ? 'bg-orange-950/40 text-orange-200'
                    : 'hover:bg-slate-800/60 text-slate-300',
                ].join(' ')}
              >
                <td className="px-4 py-2.5 font-mono text-slate-400">{i + 1}</td>
                <td className="px-4 py-2.5">
                  <span
                    className={[
                      'font-semibold',
                      det.confidence >= 0.9
                        ? 'text-green-400'
                        : det.confidence >= 0.7
                        ? 'text-yellow-400'
                        : 'text-red-400',
                    ].join(' ')}
                  >
                    {(det.confidence * 100).toFixed(1)}%
                  </span>
                </td>
                <td className="px-4 py-2.5 font-mono">
                  {det.streak_length_px != null ? det.streak_length_px.toFixed(0) : '—'}
                </td>
                <td className="px-4 py-2.5 font-mono">
                  {det.ra_deg != null ? det.ra_deg.toFixed(4) : '—'}
                </td>
                <td className="px-4 py-2.5 font-mono">
                  {det.dec_deg != null ? det.dec_deg.toFixed(4) : '—'}
                </td>
                <td className="px-4 py-2.5">
                  {best ? (
                    <span className="text-yellow-300">{best.satellite_name ?? `NORAD ${best.norad_id}`}</span>
                  ) : (
                    <span className="text-slate-600">—</span>
                  )}
                </td>
                <td className="px-4 py-2.5">
                  {best?.confidence != null ? (
                    <span className="text-slate-400">{(best.confidence * 100).toFixed(0)}%</span>
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
  )
}
