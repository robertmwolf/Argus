import { useEffect, useState } from 'react'

const STATUS_LABEL = {
  queued:     'Queued — waiting for worker',
  processing: 'Analyzing streaks…',
}

const CORNER_CLASSES = [
  'top-3 left-3 border-t-2 border-l-2',
  'top-3 right-3 border-t-2 border-r-2',
  'bottom-3 left-3 border-b-2 border-l-2',
  'bottom-3 right-3 border-b-2 border-r-2',
]

/**
 * ProcessingView — shows a raw image preview with a scan-line animation
 * while the pipeline is running.
 *
 * Props:
 *   jobId  — UUID of the queued/processing job
 *   status — 'queued' | 'processing'
 */
export default function ProcessingView({ jobId, status }) {
  const [previewOk, setPreviewOk] = useState(false)
  const [previewSrc, setPreviewSrc] = useState(null)

  useEffect(() => {
    if (!jobId) return
    // Cache-bust so we always get the latest (handles cases where
    // preview is generated slightly after the upload response).
    setPreviewSrc(`/api/preview/${jobId}?t=${Date.now()}`)
    setPreviewOk(false)
  }, [jobId])

  return (
    <div className="relative overflow-hidden rounded-xl bg-slate-900 border border-slate-800">
      {previewSrc && (
        <img
          src={previewSrc}
          alt="Raw image preview"
          className={`w-full rounded-xl transition-opacity duration-500 ${previewOk ? 'opacity-70' : 'opacity-0 absolute'}`}
          onLoad={() => setPreviewOk(true)}
          onError={() => setPreviewSrc(null)}
        />
      )}

      {!previewOk && (
        <div className="flex items-center justify-center h-52 text-slate-600 text-sm gap-2">
          <svg className="w-5 h-5 animate-spin text-slate-600" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
          </svg>
          {status === 'queued' ? 'Waiting in queue…' : 'Loading preview…'}
        </div>
      )}

      {/* Scan sweep — only shown when preview is visible */}
      {previewOk && (
        <div className="absolute inset-0 overflow-hidden rounded-xl pointer-events-none">
          <div
            className="animate-scan absolute inset-x-0"
            style={{
              height: '28%',
              background: [
                'linear-gradient(to bottom,',
                'transparent 0%,',
                'rgba(0,220,255,0.06) 20%,',
                'rgba(0,220,255,0.18) 50%,',
                'rgba(0,220,255,0.06) 80%,',
                'transparent 100%)',
              ].join(' '),
              boxShadow: '0 0 30px 4px rgba(0,220,255,0.12)',
            }}
          />
        </div>
      )}

      {/* Corner markers */}
      {previewOk && CORNER_CLASSES.map((cls, i) => (
        <div
          key={i}
          className={`absolute w-5 h-5 border-cyan-500/70 pointer-events-none ${cls}`}
        />
      ))}

      {/* Status badge */}
      <div className="absolute top-3 right-3 bg-slate-950/80 backdrop-blur-sm border border-cyan-500/40 rounded-lg px-3 py-1.5 flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
        <span className="text-xs font-medium text-cyan-300">
          {STATUS_LABEL[status] ?? status}
        </span>
      </div>
    </div>
  )
}
