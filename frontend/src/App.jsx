import { useState } from 'react'
import DetectionTable from './components/DetectionTable'
import ResultViewer from './components/ResultViewer'
import UploadZone from './components/UploadZone'

export default function App() {
  const [result, setResult] = useState(null)   // {jobId, detections, ...}
  const [error, setError] = useState(null)
  const [highlightIndex, setHighlightIndex] = useState(null)

  const handleResult = (data) => {
    setError(null)
    setHighlightIndex(null)
    setResult(data)
  }

  const handleReset = () => {
    setResult(null)
    setError(null)
    setHighlightIndex(null)
  }

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
        {result && (
          <button
            onClick={handleReset}
            className="ml-auto text-xs text-slate-400 hover:text-white border border-slate-700 hover:border-slate-500 rounded-lg px-3 py-1.5 transition-colors"
          >
            New upload
          </button>
        )}
      </header>

      <main className="max-w-5xl mx-auto px-6 py-8 flex flex-col gap-6">
        {!result && (
          <>
            <div className="text-center mb-2">
              <p className="text-slate-400 text-sm">
                Upload a FITS or PNG telescope image to detect and identify satellite streaks.
              </p>
            </div>
            <UploadZone onResult={handleResult} onError={setError} />
          </>
        )}

        {error && (
          <div className="rounded-xl bg-red-950/40 border border-red-800 px-4 py-3 text-sm text-red-300">
            {error}
          </div>
        )}

        {result && (
          <>
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-base font-semibold text-white">{result.filename}</h2>
                <p className="text-xs text-slate-500 mt-0.5">
                  {result.detections.length} detection{result.detections.length !== 1 ? 's' : ''}
                  {result.obs_epoch ? ` · ${result.obs_epoch}` : ''}
                </p>
              </div>
            </div>

            <ResultViewer
              jobId={result.jobId}
              detections={result.detections}
              highlightIndex={highlightIndex}
              onHover={setHighlightIndex}
            />

            <DetectionTable
              detections={result.detections}
              highlightIndex={highlightIndex}
              onRowClick={setHighlightIndex}
            />
          </>
        )}
      </main>
    </div>
  )
}
