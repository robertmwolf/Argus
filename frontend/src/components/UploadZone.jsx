import { useCallback, useState } from 'react'

const ACCEPTED = '.fits,.fit,.fts,.png'
const ACCEPTED_SET = new Set(['.fits', '.fit', '.fts', '.png'])
const POLL_INTERVAL_MS = 2000

function ext(filename) {
  const i = filename.lastIndexOf('.')
  return i >= 0 ? filename.slice(i).toLowerCase() : ''
}

/**
 * UploadZone — drag-and-drop FITS/PNG upload with polling.
 *
 * Props:
 *   onResult(resultPayload) — called when the job reaches "complete"
 *   onError(message)        — called on network or processing errors
 */
export default function UploadZone({ onResult, onError }) {
  const [dragging, setDragging] = useState(false)
  const [status, setStatus] = useState(null) // null | 'uploading' | 'queued' | 'processing' | 'complete' | 'failed'
  const [filename, setFilename] = useState(null)

  const processFile = useCallback(async (file) => {
    if (!ACCEPTED_SET.has(ext(file.name))) {
      onError?.(`Unsupported file type. Accepted: ${ACCEPTED}`)
      return
    }

    setFilename(file.name)
    setStatus('uploading')

    const form = new FormData()
    form.append('file', file)

    let jobId
    try {
      const res = await fetch('/api/upload', { method: 'POST', body: form })
      if (res.status === 413) throw new Error('File exceeds 100 MB limit')
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail ?? `Upload failed (${res.status})`)
      }
      const data = await res.json()
      jobId = data.job_id
      setStatus('queued')
    } catch (err) {
      setStatus(null)
      onError?.(err.message)
      return
    }

    // Poll until complete or failed
    const poll = setInterval(async () => {
      try {
        const res = await fetch(`/api/result/${jobId}`)
        if (!res.ok) return
        const data = await res.json()
        setStatus(data.status)
        if (data.status === 'complete') {
          clearInterval(poll)
          onResult?.({ ...data, jobId })
        } else if (data.status === 'failed') {
          clearInterval(poll)
          onError?.('Processing failed on the server')
        }
      } catch {
        // transient network error — keep polling
      }
    }, POLL_INTERVAL_MS)
  }, [onResult, onError])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) processFile(file)
  }, [processFile])

  const onInputChange = useCallback((e) => {
    const file = e.target.files[0]
    if (file) processFile(file)
    e.target.value = ''
  }, [processFile])

  const statusLabel = {
    uploading: 'Uploading…',
    queued: 'Queued — waiting for worker',
    processing: 'Processing…',
    complete: 'Complete',
    failed: 'Failed',
  }[status] ?? 'Drop a FITS or PNG file here'

  const statusColour = {
    uploading: 'text-blue-400',
    queued: 'text-yellow-400',
    processing: 'text-blue-400 animate-pulse',
    complete: 'text-green-400',
    failed: 'text-red-400',
  }[status] ?? 'text-slate-400'

  return (
    <div
      className={[
        'relative flex flex-col items-center justify-center gap-3',
        'rounded-2xl border-2 border-dashed transition-colors',
        'min-h-48 p-8 cursor-pointer select-none',
        dragging
          ? 'border-cyan-400 bg-cyan-950/30'
          : 'border-slate-600 hover:border-slate-400 bg-slate-900/40',
      ].join(' ')}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      onClick={() => document.getElementById('argus-file-input').click()}
    >
      <input
        id="argus-file-input"
        type="file"
        accept={ACCEPTED}
        className="hidden"
        onChange={onInputChange}
      />

      {/* Icon */}
      <svg className="w-10 h-10 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
          d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
      </svg>

      {filename && status !== null ? (
        <p className="text-sm text-slate-300 font-medium">{filename}</p>
      ) : (
        <p className="text-sm text-slate-400">
          Drag &amp; drop, or <span className="text-cyan-400 underline">browse</span>
        </p>
      )}

      <p className={`text-xs font-medium ${statusColour}`}>{statusLabel}</p>

      {status && status !== 'complete' && status !== 'failed' && (
        <div className="w-full max-w-xs h-1 rounded-full bg-slate-700 overflow-hidden">
          <div className="h-full bg-cyan-500 rounded-full animate-pulse w-1/2" />
        </div>
      )}
    </div>
  )
}
