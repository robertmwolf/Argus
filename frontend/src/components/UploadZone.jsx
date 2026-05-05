import { useCallback, useState } from 'react'

const ACCEPTED = '.fits,.fit,.fts,.png'
const ACCEPTED_SET = new Set(['.fits', '.fit', '.fts', '.png'])

function ext(filename) {
  const i = filename.lastIndexOf('.')
  return i >= 0 ? filename.slice(i).toLowerCase() : ''
}

/**
 * UploadZone — drag-and-drop FITS/PNG upload.
 *
 * Props:
 *   onQueued(jobId, filename) — called once the file is uploaded and queued
 *   onError(message)          — called on network or validation errors
 */
export default function UploadZone({ onQueued, onError }) {
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)

  const processFile = useCallback(async (file) => {
    if (!ACCEPTED_SET.has(ext(file.name))) {
      onError?.(`Unsupported file type. Accepted: ${ACCEPTED}`)
      return
    }

    setUploading(true)
    const form = new FormData()
    form.append('file', file)

    try {
      const res = await fetch('/api/upload', { method: 'POST', body: form })
      if (res.status === 413) throw new Error('File exceeds 100 MB limit')
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail ?? `Upload failed (${res.status})`)
      }
      const data = await res.json()
      onQueued?.(data.job_id, file.name)
    } catch (err) {
      onError?.(err.message)
      setUploading(false)
    }
  }, [onQueued, onError])

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

  return (
    <div
      className={[
        'relative flex flex-col items-center justify-center gap-3',
        'rounded-2xl border-2 border-dashed transition-colors',
        'min-h-48 p-8 cursor-pointer select-none',
        uploading
          ? 'border-blue-500 bg-blue-950/20 pointer-events-none'
          : dragging
          ? 'border-cyan-400 bg-cyan-950/30'
          : 'border-slate-600 hover:border-slate-400 bg-slate-900/40',
      ].join(' ')}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      onClick={() => !uploading && document.getElementById('argus-file-input').click()}
    >
      <input
        id="argus-file-input"
        type="file"
        accept={ACCEPTED}
        className="hidden"
        onChange={onInputChange}
      />

      {uploading ? (
        <>
          <svg className="w-10 h-10 text-blue-400 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
          </svg>
          <p className="text-sm text-blue-400 font-medium">Uploading…</p>
        </>
      ) : (
        <>
          <svg className="w-10 h-10 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
          </svg>
          <p className="text-sm text-slate-400">
            Drag &amp; drop, or <span className="text-cyan-400 underline">browse</span>
          </p>
          <p className="text-xs text-slate-500">FITS · FIT · FTS · PNG · up to 100 MB</p>
        </>
      )}
    </div>
  )
}
