import { useEffect, useState } from 'react'

/**
 * DetectorsPanel — shows all available detectors with interactive toggles.
 *
 * Props:
 *   onSelectionChange(Set<string>) — called whenever the enabled set changes
 */
export default function DetectorsPanel({ onSelectionChange }) {
  const [detectors, setDetectors] = useState(null) // null = loading
  const [selected, setSelected] = useState(null)   // null until loaded
  const [fetchError, setFetchError] = useState(false)

  useEffect(() => {
    fetch('/api/detectors')
      .then(r => r.json())
      .then(data => {
        const list = data.detectors ?? []
        setDetectors(list)
        // Default: all 'active' detectors enabled
        const initial = new Set(list.filter(d => d.status === 'active').map(d => d.id))
        setSelected(initial)
        onSelectionChange?.(initial)
      })
      .catch(() => setFetchError(true))
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const toggle = (id) => {
    setSelected(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      onSelectionChange?.(next)
      return next
    })
  }

  const selectAll = () => {
    if (!detectors) return
    const next = new Set(detectors.filter(d => d.status === 'active').map(d => d.id))
    setSelected(next)
    onSelectionChange?.(next)
  }

  const selectNone = () => {
    const next = new Set()
    setSelected(next)
    onSelectionChange?.(next)
  }

  if (fetchError) {
    return (
      <div className="rounded-xl border border-slate-800 bg-slate-900/40 px-4 py-3 text-xs text-slate-500 text-center">
        Detector status unavailable
      </div>
    )
  }

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/40">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800">
        <span className="text-sm font-semibold text-slate-200">Active Detectors</span>
        {selected !== null && (
          <div className="flex items-center gap-2">
            <button
              onClick={selectAll}
              className="text-xs text-slate-400 hover:text-cyan-400 transition-colors"
            >
              All
            </button>
            <span className="text-slate-700">·</span>
            <button
              onClick={selectNone}
              className="text-xs text-slate-400 hover:text-slate-200 transition-colors"
            >
              None
            </button>
          </div>
        )}
      </div>

      {/* Table */}
      {detectors === null ? (
        <div className="px-4 py-6 flex justify-center">
          <svg className="w-5 h-5 text-slate-600 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
          </svg>
        </div>
      ) : (
        <table className="w-full text-xs">
          <thead>
            <tr className="text-slate-500 border-b border-slate-800/60">
              <th className="text-left px-4 py-2 font-medium w-8"></th>
              <th className="text-left px-2 py-2 font-medium">Model</th>
              <th className="text-left px-2 py-2 font-medium hidden sm:table-cell">Type</th>
              <th className="text-left px-2 py-2 font-medium hidden md:table-cell">Dataset</th>
              <th className="text-left px-2 py-2 font-medium">Status</th>
            </tr>
          </thead>
          <tbody>
            {detectors.map((det, i) => {
              const isActive = det.status === 'active'
              const isChecked = selected?.has(det.id) ?? false
              return (
                <tr
                  key={det.id}
                  className={[
                    'border-b border-slate-800/40 last:border-0 transition-colors',
                    isActive ? 'cursor-pointer hover:bg-slate-800/30' : 'opacity-50',
                  ].join(' ')}
                  onClick={() => isActive && toggle(det.id)}
                >
                  {/* Checkbox */}
                  <td className="px-4 py-2.5">
                    <input
                      type="checkbox"
                      checked={isChecked}
                      disabled={!isActive}
                      onChange={() => isActive && toggle(det.id)}
                      onClick={e => e.stopPropagation()}
                      className="w-3.5 h-3.5 accent-cyan-500 cursor-pointer disabled:cursor-default"
                    />
                  </td>

                  {/* Name */}
                  <td className="px-2 py-2.5 text-slate-200 font-medium">{det.name}</td>

                  {/* Type badge */}
                  <td className="px-2 py-2.5 hidden sm:table-cell">
                    <span className={[
                      'inline-block px-1.5 py-0.5 rounded text-[10px] font-medium',
                      det.type === 'ml'
                        ? 'bg-violet-950/60 text-violet-400 border border-violet-800/50'
                        : 'bg-slate-800/60 text-slate-400 border border-slate-700/50',
                    ].join(' ')}>
                      {det.type === 'ml' ? 'ML' : 'Classical'}
                    </span>
                  </td>

                  {/* Dataset */}
                  <td className="px-2 py-2.5 text-slate-500 hidden md:table-cell">
                    {det.dataset || '—'}
                  </td>

                  {/* Status */}
                  <td className="px-2 py-2.5">
                    <StatusDot status={det.status} />
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
    </div>
  )
}

function StatusDot({ status }) {
  const cfg = {
    active:      { dot: 'bg-green-400', label: 'Active',      text: 'text-green-400' },
    no_weights:  { dot: 'bg-amber-400',  label: 'No weights',  text: 'text-amber-400' },
    unavailable: { dot: 'bg-slate-500',  label: 'Unavailable', text: 'text-slate-500' },
  }[status] ?? { dot: 'bg-slate-600', label: status, text: 'text-slate-500' }

  return (
    <span className={`inline-flex items-center gap-1.5 ${cfg.text}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${cfg.dot}`} />
      {cfg.label}
    </span>
  )
}
