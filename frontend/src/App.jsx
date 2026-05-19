import { useEffect, useMemo, useState } from 'react'
import DetectionTable from './components/DetectionTable'
import FilterPanel from './components/FilterPanel'
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
  const [modelLabel, setModelLabel] = useState(null)
  const [spaceTrackDataRefreshedAt, setSpaceTrackDataRefreshedAt] = useState(null)
  const [headerObsEpoch, setHeaderObsEpoch] = useState(null)
  const [disabledStreaks, setDisabledStreaks] = useState(new Set())   // Set of displayDetections indices
  const [methodThresholds, setMethodThresholds] = useState({})        // { method: 0-1 }

  useEffect(() => {
    fetch('/health')
      .then(r => r.json())
      .then(d => {
        setModelLabel(d.model_label)
        setSpaceTrackDataRefreshedAt(d.space_track_data_refreshed_at ?? null)
      })
      .catch(() => {})
  }, [])

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

  useEffect(() => {
    if (!jobId || jobStatus !== 'complete') return

    let cancelled = false
    const refreshCompleteResult = async () => {
      try {
        const res = await fetch(`/api/result/${jobId}`)
        if (!res.ok) return
        const data = await res.json()
        if (!cancelled && data.status === 'complete') {
          setResult({ ...data, jobId })
        }
      } catch {
        // completed-result refresh is best-effort
      }
    }

    refreshCompleteResult()
    return () => {
      cancelled = true
    }
  }, [jobId, jobStatus])

  useEffect(() => {
    if (!jobId || jobStatus !== 'complete') {
      setHeaderObsEpoch(null)
      return
    }

    let cancelled = false
    const loadHeaderObsEpoch = async () => {
      try {
        const res = await fetch(`/api/fits-header/${jobId}`)
        if (!res.ok) return
        const data = await res.json()
        const dateObs = data.cards?.find((card) => card.key === 'DATE-OBS')?.value
        if (!cancelled && dateObs) {
          setHeaderObsEpoch(normaliseDateObs(dateObs))
        }
      } catch {
        // header date is a display fallback only
      }
    }

    loadHeaderObsEpoch()
    return () => {
      cancelled = true
    }
  }, [jobId, jobStatus])

  const handleThresholdChange = (method, value) => {
    setMethodThresholds(prev => ({ ...prev, [method]: value }))
  }

  const handleToggleStreak = (idx) => {
    setDisabledStreaks(prev => {
      const next = new Set(prev)
      if (next.has(idx)) next.delete(idx)
      else next.add(idx)
      return next
    })
  }

  const handleQueued = (newJobId, newFilename) => {
    setError(null)
    setHighlightIndex(null)
    setResult(null)
    setHeaderObsEpoch(null)
    setDisabledStreaks(new Set())
    setMethodThresholds({})
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
    setHeaderObsEpoch(null)
    setDisabledStreaks(new Set())
    setMethodThresholds({})
  }

  const isProcessing = jobId && jobStatus && jobStatus !== 'complete' && jobStatus !== 'failed'
  const isComplete = result !== null

  // One entry per streak, sorted by primary confidence descending.
  const displayDetections = useMemo(() => {
    if (!result?.detections) return []
    return [...result.detections].sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0))
  }, [result?.detections])

  // Streaks that pass both the per-method threshold and are not manually disabled.
  const visibleSet = useMemo(() => {
    const set = new Set()
    displayDetections.forEach((det, idx) => {
      if (disabledStreaks.has(idx)) return
      const sources = det.sources ?? [{ method: det.method, confidence: det.confidence }]
      const passes = sources.every(s => (s.confidence ?? 1) >= (methodThresholds[s.method] ?? 0))
      if (passes) set.add(idx)
    })
    return set
  }, [displayDetections, disabledStreaks, methodThresholds])

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
        {modelLabel && (
          <span className="ml-1 inline-flex items-center gap-1.5 text-xs font-medium px-2 py-0.5 rounded-full border border-cyan-700/50 bg-cyan-950/30 text-cyan-400">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
            {modelLabel}
          </span>
        )}
        <div className="ml-auto flex items-center gap-3">
          <div className="hidden sm:flex flex-col items-end leading-tight">
            <span className="text-[10px] uppercase tracking-wide text-slate-500">
              Space-Track Data Refreshed At
            </span>
            <span className="font-mono text-xs text-slate-300">
              {formatHeaderDateTime(spaceTrackDataRefreshedAt)}
            </span>
          </div>
          {(isProcessing || isComplete) && (
            <button
              onClick={handleReset}
              className="text-xs text-slate-400 hover:text-white border border-slate-700 hover:border-slate-500 rounded-lg px-3 py-1.5 transition-colors"
            >
              New upload
            </button>
          )}
        </div>
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
                {displayDetections.length} streak{displayDetections.length !== 1 ? 's' : ''} detected
                {result.obs_epoch ? ` · ${result.obs_epoch}` : ''}
              </span>
            </div>

            <FilterPanel
              detections={displayDetections}
              methodThresholds={methodThresholds}
              onThresholdChange={handleThresholdChange}
              visibleCount={visibleSet.size}
            />

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
              <div className="lg:col-span-2">
                <ResultViewer
                  jobId={result.jobId}
                  detections={displayDetections}
                  visibleSet={visibleSet}
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
              detections={displayDetections}
              visibleSet={visibleSet}
              highlightIndex={highlightIndex}
              onRowClick={setHighlightIndex}
              onToggleStreak={handleToggleStreak}
              photoTakenAt={result.obs_epoch ?? headerObsEpoch}
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

function normaliseDateObs(value) {
  if (!value) return null
  const raw = String(value).trim().replace(' ', 'T')
  if (/[zZ]$|[+-]\d{2}:?\d{2}$/.test(raw)) return raw
  return `${raw}Z`
}

function formatHeaderDateTime(value) {
  if (!value) return 'Not available'
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
