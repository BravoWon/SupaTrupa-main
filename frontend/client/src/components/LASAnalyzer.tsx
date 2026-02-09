import { useState, useCallback, useRef, useEffect } from 'react';
import { ChevronLeft, ChevronRight, Play, Pause } from 'lucide-react';
import { cn } from '@/lib/utils';
import { getApiUrl } from '@/lib/api';
import type { LASFileState, LASAnalyzeWindowResponse, AnalyzerHistoryEntry, WindowedSignature } from '@/types';

interface LASAnalyzerProps {
  fileState: LASFileState;
  curveMapping: Record<string, string>;
  onWindowData: (
    records: Record<string, number>[],
    regime: LASAnalyzeWindowResponse['regime'],
    windowedSigs: WindowedSignature[]
  ) => void;
  onHistoryUpdate: (history: AnalyzerHistoryEntry[]) => void;
  className?: string;
}

export function LASAnalyzer({
  fileState,
  curveMapping,
  onWindowData,
  onHistoryUpdate,
  className,
}: LASAnalyzerProps) {
  const [windowSize, setWindowSize] = useState(200);
  const [stepSize, setStepSize] = useState(100);
  const [windowStart, setWindowStart] = useState(fileState.index_min);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playSpeed, setPlaySpeed] = useState(1000);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState<AnalyzerHistoryEntry[]>([]);
  void history;
  const playRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const windowEnd = Math.min(windowStart + windowSize, fileState.index_max);
  const range = fileState.index_max - fileState.index_min;

  const fetchWindow = useCallback(
    async (start: number, end: number) => {
      if (Object.keys(curveMapping).length === 0) return;
      setLoading(true);
      try {
        const res = await fetch(getApiUrl(`/api/v1/las/${fileState.file_id}/analyze-window`), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            curve_mapping: curveMapping,
            start_index: start,
            end_index: end,
            max_points: 2000,
            tda_window_size: 20,
            tda_stride: 5,
          }),
        });
        if (res.ok) {
          const data: LASAnalyzeWindowResponse = await res.json();
          onWindowData(data.records, data.regime, data.windowed_signatures);

          if (data.regime) {
            setHistory((prev) => {
              const entry: AnalyzerHistoryEntry = {
                start,
                end,
                regime: data.regime!.regime,
                confidence: data.regime!.confidence,
                color: data.regime!.color,
              };
              const next = [...prev, entry];
              onHistoryUpdate(next);
              return next;
            });
          }
        }
      } catch {
        /* ignore */
      } finally {
        setLoading(false);
      }
    },
    [fileState.file_id, curveMapping, onWindowData, onHistoryUpdate]
  );

  const debouncedFetch = useCallback(
    (start: number) => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        const end = Math.min(start + windowSize, fileState.index_max);
        fetchWindow(start, end);
      }, 300);
    },
    [fetchWindow, windowSize, fileState.index_max]
  );

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setWindowStart(val);
    debouncedFetch(val);
  };

  const stepForward = useCallback(() => {
    setWindowStart((prev) => {
      const next = Math.min(prev + stepSize, fileState.index_max - windowSize);
      const end = Math.min(next + windowSize, fileState.index_max);
      fetchWindow(next, end);
      return next;
    });
  }, [stepSize, windowSize, fileState.index_max, fetchWindow]);

  const stepBackward = useCallback(() => {
    setWindowStart((prev) => {
      const next = Math.max(prev - stepSize, fileState.index_min);
      const end = Math.min(next + windowSize, fileState.index_max);
      fetchWindow(next, end);
      return next;
    });
  }, [stepSize, windowSize, fileState.index_min, fileState.index_max, fetchWindow]);

  // Auto-play
  useEffect(() => {
    if (isPlaying) {
      playRef.current = setInterval(() => {
        setWindowStart((prev) => {
          const next = prev + stepSize;
          if (next >= fileState.index_max - windowSize) {
            setIsPlaying(false);
            return prev;
          }
          const end = Math.min(next + windowSize, fileState.index_max);
          fetchWindow(next, end);
          return next;
        });
      }, playSpeed);
    }
    return () => {
      if (playRef.current) clearInterval(playRef.current);
    };
  }, [isPlaying, playSpeed, stepSize, windowSize, fileState.index_max, fetchWindow]);

  const sliderPct = range > 0 ? ((windowStart - fileState.index_min) / range) * 100 : 0;
  const windowPct = range > 0 ? (windowSize / range) * 100 : 100;

  return (
    <div className={cn('flex flex-col gap-1 px-3 py-1', className)}>
      {/* Top row: controls */}
      <div className="flex items-center gap-2">
        {/* Step buttons */}
        <button
          onClick={stepBackward}
          disabled={windowStart <= fileState.index_min}
          className="w-6 h-6 flex items-center justify-center border border-border hover:bg-primary/20 disabled:opacity-30 text-foreground"
        >
          <ChevronLeft className="w-3.5 h-3.5" />
        </button>

        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className={cn(
            'w-6 h-6 flex items-center justify-center border border-border hover:bg-primary/20',
            isPlaying && 'bg-primary/20 text-primary'
          )}
        >
          {isPlaying ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
        </button>

        <button
          onClick={stepForward}
          disabled={windowStart >= fileState.index_max - windowSize}
          className="w-6 h-6 flex items-center justify-center border border-border hover:bg-primary/20 disabled:opacity-30 text-foreground"
        >
          <ChevronRight className="w-3.5 h-3.5" />
        </button>

        {/* Depth display */}
        <span className="text-[10px] font-mono text-primary min-w-[120px]">
          {windowStart.toFixed(0)} – {windowEnd.toFixed(0)} {fileState.index_unit}
        </span>

        {/* Slider */}
        <div className="flex-1 relative h-4">
          {/* Window highlight bar */}
          <div
            className="absolute top-1 h-2 bg-primary/20 border border-primary/30 pointer-events-none"
            style={{ left: `${sliderPct}%`, width: `${Math.min(windowPct, 100 - sliderPct)}%` }}
          />
          <input
            type="range"
            min={fileState.index_min}
            max={fileState.index_max - windowSize}
            step={1}
            value={windowStart}
            onChange={handleSliderChange}
            className="w-full h-4 appearance-none bg-transparent cursor-pointer [&::-webkit-slider-runnable-track]:h-0.5 [&::-webkit-slider-runnable-track]:bg-border [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-2 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:bg-primary [&::-webkit-slider-thumb]:-mt-[7px]"
          />
        </div>

        {/* Window / Step / Speed controls */}
        <div className="flex items-center gap-1.5 shrink-0">
          <label className="text-[9px] font-mono text-muted-foreground">WIN</label>
          <input
            type="number"
            value={windowSize}
            onChange={(e) => setWindowSize(Math.max(10, parseInt(e.target.value) || 10))}
            className="w-14 h-5 text-[10px] font-mono bg-card border border-border px-1 text-foreground"
          />
          <label className="text-[9px] font-mono text-muted-foreground">STEP</label>
          <input
            type="number"
            value={stepSize}
            onChange={(e) => setStepSize(Math.max(1, parseInt(e.target.value) || 1))}
            className="w-14 h-5 text-[10px] font-mono bg-card border border-border px-1 text-foreground"
          />
          <label className="text-[9px] font-mono text-muted-foreground">SPD</label>
          <select
            value={playSpeed}
            onChange={(e) => setPlaySpeed(parseInt(e.target.value))}
            className="h-5 text-[9px] font-mono bg-card border border-border px-0.5 text-foreground"
          >
            <option value={2000}>0.5x</option>
            <option value={1000}>1x</option>
            <option value={500}>2x</option>
            <option value={250}>4x</option>
          </select>
        </div>

        {loading && (
          <div className="w-3 h-3 border border-primary border-t-transparent rounded-full animate-spin shrink-0" />
        )}
      </div>
    </div>
  );
}
