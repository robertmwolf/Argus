import { useEffect, useState } from 'react'

// Fields shown in the collapsed summary view, in display order
const SUMMARY_KEYS = [
  'DATE-OBS', 'DATE', 'OBJECT', 'EXPTIME', 'FILTER',
  'TELESCOP', 'INSTRUME', 'OBSERVER', 'NAXIS1', 'NAXIS2',
  'RA', 'DEC', 'AIRMASS', 'ORIGIN',
]

function formatValue(key, value) {
  if (value === null || value === undefined) return '—'
  if (key === 'EXPTIME') return `${value} s`
  return String(value)
}

/**
 * FitsHeaderPanel — collapsible FITS primary header viewer.
 *
 * Props:
 *   jobId — UUID of the observation whose header to display
 */
export default function FitsHeaderPanel({ jobId }) {
  const [cards, setCards] = useState(null)   // null = loading, [] = no header
  const [expanded, setExpanded] = useState(false)
  const [search, setSearch] = useState('')

  useEffect(() => {
    if (!jobId) return
    let cancelled = false
    fetch(`/api/fits-header/${jobId}`)
      .then((r) => r.json())
      .then((data) => { if (!cancelled) setCards(data.cards ?? []) })
      .catch(() => { if (!cancelled) setCards([]) })
    return () => { cancelled = true }
  }, [jobId])

  if (cards === null) {
    return (
      <div className="rounded-xl border border-slate-700 bg-slate-900/50 p-4 flex items-center gap-2 text-slate-500 text-sm">
        <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
        </svg>
        Loading header…
      </div>
    )
  }

  if (cards.length === 0) {
    return (
      <div className="rounded-xl border border-slate-700 bg-slate-900/50 p-4 text-slate-500 text-sm">
        No FITS header available for this file.
      </div>
    )
  }

  const summaryCards = SUMMARY_KEYS
    .map((k) => cards.find((c) => c.key === k))
    .filter(Boolean)

  const filteredCards = expanded
    ? search.trim()
      ? cards.filter((c) =>
          c.key.toLowerCase().includes(search.toLowerCase()) ||
          String(c.value ?? '').toLowerCase().includes(search.toLowerCase()) ||
          (c.comment ?? '').toLowerCase().includes(search.toLowerCase())
        )
      : cards
    : summaryCards

  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900/50 overflow-hidden flex flex-col">
      {/* Panel header */}
      <div className="px-4 py-3 border-b border-slate-700 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <svg className="w-4 h-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M9 12h3.75M9 15h3.75M9 18h3.75m3 .75H18a2.25 2.25 0 002.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424 48.424 0 00-1.123-.08m-5.801 0c-.065.21-.1.433-.1.664 0 .414.336.75.75.75h4.5a.75.75 0 00.75-.75 2.25 2.25 0 00-.1-.664m-5.8 0A2.251 2.251 0 0113.5 2.25H15c1.012 0 1.867.668 2.15 1.586m-5.8 0c-.376.023-.75.05-1.124.08C9.095 4.01 8.25 4.973 8.25 6.108V8.25m0 0H4.875c-.621 0-1.125.504-1.125 1.125v11.25c0 .621.504 1.125 1.125 1.125h9.75c.621 0 1.125-.504 1.125-1.125V9.375c0-.621-.504-1.125-1.125-1.125H8.25zM6.75 12h.008v.008H6.75V12zm0 3h.008v.008H6.75V15zm0 3h.008v.008H6.75V18z" />
          </svg>
          <h3 className="text-sm font-semibold text-slate-300">FITS Header</h3>
        </div>
        <span className="text-xs text-slate-500">{cards.length} keys</span>
      </div>

      {/* Search box — only when expanded */}
      {expanded && (
        <div className="px-3 py-2 border-b border-slate-700">
          <input
            type="text"
            placeholder="Search keys, values, comments…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full bg-slate-800 text-slate-200 text-xs px-3 py-1.5 rounded-lg border border-slate-600 focus:outline-none focus:border-cyan-500 placeholder-slate-600"
          />
        </div>
      )}

      {/* Card table */}
      <div className={`overflow-y-auto ${expanded ? 'max-h-72' : 'max-h-56'}`}>
        {filteredCards.length === 0 ? (
          <p className="px-4 py-4 text-xs text-slate-500 text-center">No matching keys.</p>
        ) : (
          <table className="w-full text-xs">
            <tbody>
              {filteredCards.map((card, i) => (
                <tr key={i} className="border-b border-slate-800/60 hover:bg-slate-800/40 group">
                  <td className="px-3 py-1.5 font-mono text-cyan-400 font-semibold whitespace-nowrap align-top w-24">
                    {card.key}
                  </td>
                  <td className="px-3 py-1.5 font-mono text-slate-200 break-all align-top">
                    {formatValue(card.key, card.value)}
                  </td>
                  {card.comment && (
                    <td className="px-3 py-1.5 text-slate-500 align-top hidden xl:table-cell max-w-[12rem] truncate">
                      {card.comment}
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Expand / collapse toggle */}
      <button
        onClick={() => { setExpanded((v) => !v); setSearch('') }}
        className="w-full py-2 text-xs text-slate-400 hover:text-slate-200 border-t border-slate-700 transition-colors flex items-center justify-center gap-1"
      >
        <svg
          className={`w-3.5 h-3.5 transition-transform ${expanded ? 'rotate-180' : ''}`}
          fill="none" viewBox="0 0 24 24" stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
        {expanded ? 'Show summary' : `Show all ${cards.length} header fields`}
      </button>
    </div>
  )
}
