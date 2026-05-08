import { useEffect, useState } from 'react'
import DetectionTable from './components/DetectionTable'
import FitsHeaderPanel from './components/FitsHeaderPanel'
import IdentificationPanel from './components/IdentificationPanel'
import ProcessingView from './components/ProcessingView'
import ResultViewer from './components/ResultViewer'
import UploadZone from './components/UploadZone'

const POLL_INTERVAL_MS = 2000

export default function App() {
  const [jobId, setJobId] = useState(null)
  const [filename, setFilename] = useState(null)
  const [jobStatus, setJobStatus] = useState(null) // queued | processing | complete | failed
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [highlightIndex, setHighlightIndex] = useState(null)

  // Poll for job completion once we have a jobId
  useEffect(() => {
    if (!jobId || jobStatus === 'complete' || jobStatus === 'failed') return

    let cancelled = false

    const pollOnce = async () => {
      try {
        const res = await fetch(`/api/result/${jobId}`)
        if (!res.ok) return
        const data = await res.json()
        if (cancelled) return
        setJobStatus(data.status)
        if (data.status === 'complete') {
          setResult({ ...data, jobId })
        } else if (data.status === 'failed') {
          setError('Processing failed on the server.')
        }
      } catch {
        // transient network error — keep polling
      }
    }

    pollOnce()
    const poll = setInterval(pollOnce, POLL_INTERVAL_MS)

    return () => {
      cancelled = true
      clearInterval(poll)
    }
  }, [jobId, jobStatus])

  const handleQueued = (newJobId, newFilename) => {
    setError(null)
    setHighlightIndex(null)
    setResult(null)
    setJobId(newJobId)
    setFilename(newFilename)
    setJobStatus('queued')
  }

  const handleReset = () => {
    setJobId(null)
    setFilename(null)
    setJobStatus(null)
    setResult(null)
    setError(null)
    setHighlightIndex(null)
  }

  const isProcessing = jobId && jobStatus && jobStatus !== 'complete' && jobStatus !== 'failed'
  const isComplete = result !== null

  return (
    <div className="min-h-screen bg-[#0d0e14] text-slate-200">
      {/* Header */}
      <header className="border-b border-slate-800 px-6 py-4 flex items-center gap-3">
        <svg className="w-6 h-6 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <circle cx="12" cy="12" r="3" strokeWidth={2} />
          <path strokeLinecap="round" strokeWidth={1.5}
            d="M12 2v2m0 16v2M2 12h2m16 0h2M4.93 4.93l1.41 1.41m11.32 11.32 1.41 1.41M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
        </svg>
        <h1 className="text-lg font-semibold tracking-tight text-white">ARGUS</h1>
        <span className="text-slate-500 text-sm">Satellite Streak Detector</span>
        {(isProcessing || isComplete) && (
          <button
            onClick={handleReset}
            className="ml-auto text-xs text-slate-400 hover:text-white border border-slate-700 hover:border-slate-500 rounded-lg px-3 py-1.5 transition-colors"
          >
            New upload
          </button>
        )}
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8 flex flex-col gap-6">
        {/* Upload zone — shown only when idle */}
        {!jobId && (
          <>
            <div className="text-center mb-2">
              <p className="text-slate-400 text-sm">
                Upload a FITS or PNG telescope image to detect and identify satellite streaks.
              </p>
            </div>
            <UploadZone onQueued={handleQueued} onError={setError} />
          </>
        )}

        {/* Error banner */}
        {error && (
          <div className="rounded-xl bg-red-950/40 border border-red-800 px-4 py-3 text-sm text-red-300">
            {error}
          </div>
        )}

        {/* Processing state — preview + scan animation + FITS header */}
        {isProcessing && (
          <>
            <div className="flex items-center gap-3">
              <h2 className="text-base font-semibold text-white truncate">{filename}</h2>
              <StatusBadge status={jobStatus} />
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
              <div className="lg:col-span-2">
                <ProcessingView jobId={jobId} status={jobStatus} />
              </div>
              <div className="lg:col-span-1">
                <FitsHeaderPanel jobId={jobId} />
              </div>
            </div>
          </>
        )}

        {/* Complete state — results */}
        {isComplete && (
          <>
            <div className="flex items-center gap-3">
              <h2 className="text-base font-semibold text-white truncate">{result.filename}</h2>
              <span className="text-xs text-slate-500">
                {result.detections.length} detection{result.detections.length !== 1 ? 's' : ''}
                {result.obs_epoch ? ` · ${result.obs_epoch}` : ''}
              </span>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
              <div className="lg:col-span-2">
                <ResultViewer
                  jobId={result.jobId}
                  detections={result.detections}
                  imageWidth={result.image_width}
                  imageHeight={result.image_height}
                  highlightIndex={highlightIndex}
                  onHover={setHighlightIndex}
                />
              </div>
              <div className="lg:col-span-1">
                <FitsHeaderPanel jobId={result.jobId} />
              </div>
            </div>

            <DetectionTable
              detections={result.detections}
              highlightIndex={highlightIndex}
              onRowClick={setHighlightIndex}
            />

            <IdentificationPanel
              detections={result.detections}
              activeIndex={highlightIndex}
            />
          </>
        )}
      </main>
    </div>
  )
}

function StatusBadge({ status }) {
  const cfg = {
    queued:     { colour: 'text-yellow-400 border-yellow-700/50 bg-yellow-950/30', dot: 'bg-yellow-400', label: 'Queued' },
    processing: { colour: 'text-cyan-400 border-cyan-700/50 bg-cyan-950/30',     dot: 'bg-cyan-400 animate-pulse', label: 'Processing…' },
    complete:   { colour: 'text-green-400 border-green-700/50 bg-green-950/30',   dot: 'bg-green-400', label: 'Complete' },
    failed:     { colour: 'text-red-400 border-red-700/50 bg-red-950/30',         dot: 'bg-red-400', label: 'Failed' },
  }[status] ?? { colour: 'text-slate-400 border-slate-700 bg-slate-900', dot: 'bg-slate-400', label: status }

  return (
    <span className={`inline-flex items-center gap-1.5 text-xs font-medium px-2.5 py-1 rounded-full border ${cfg.colour}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${cfg.dot}`} />
      {cfg.label}
    </span>
  )
}
