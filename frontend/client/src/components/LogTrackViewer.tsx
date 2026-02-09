import { useMemo, useRef, useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Layers, ZoomIn, ZoomOut } from 'lucide-react';
import { CATEGORY_COLORS, type CurveCategory } from '@/lib/lasMnemonics';
import type { LASCurveData, LASTrackConfig } from '@/types';

interface LogTrackViewerProps {
  data: LASCurveData;
  tracks?: LASTrackConfig[];
  currentDepth?: number;
  height?: number;
}

const TRACK_WIDTH = 160;
const RULER_WIDTH = 70;
const HEADER_HEIGHT = 40;
const PADDING = 8;

// Color palette for curves
const CURVE_PALETTE = [
  '#3b82f6', '#22c55e', '#ef4444', '#eab308', '#a855f7',
  '#06b6d4', '#f97316', '#ec4899', '#14b8a6', '#8b5cf6',
];

function buildDefaultTracks(data: LASCurveData): LASTrackConfig[] {
  const tracks: LASTrackConfig[] = [];
  const usedCurves = new Set<string>();

  // Group curves by category
  const byCategory = new Map<string, string[]>();
  for (const [mnem, info] of Object.entries(data.curves)) {
    const cat = info.category || 'OTHER';
    if (!byCategory.has(cat)) byCategory.set(cat, []);
    byCategory.get(cat)!.push(mnem);
  }

  // Priority order for tracks
  const priority: [string, string][] = [
    ['GAMMA', 'Gamma'],
    ['ROP', 'ROP'],
    ['WEIGHT', 'WOB / HL'],
    ['ROTARY', 'RPM'],
    ['TORQUE', 'Torque'],
    ['PRESSURE', 'SPP'],
    ['FLOW', 'Flow'],
    ['TEMPERATURE', 'Temp'],
    ['GAS', 'Gas'],
    ['DIRECTIONAL', 'Dir'],
  ];

  let trackId = 0;
  for (const [cat, label] of priority) {
    const curves = byCategory.get(cat);
    if (!curves || curves.length === 0) continue;
    const trackCurves = curves.filter(c => !usedCurves.has(c)).slice(0, 3);
    if (trackCurves.length === 0) continue;
    for (const c of trackCurves) usedCurves.add(c);
    tracks.push({
      id: `track-${trackId++}`,
      label,
      curves: trackCurves,
      color: CATEGORY_COLORS[cat as CurveCategory] || '#6b7280',
    });
  }

  // Remaining curves in an "Other" track
  const remaining = Object.keys(data.curves).filter(c => !usedCurves.has(c));
  if (remaining.length > 0) {
    tracks.push({
      id: `track-${trackId}`,
      label: 'Other',
      curves: remaining.slice(0, 4),
      color: '#6b7280',
    });
  }

  return tracks;
}

export function LogTrackViewer({ data, tracks: tracksProp, currentDepth, height = 600 }: LogTrackViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(1);
  const [scrollTop, setScrollTop] = useState(0);
  // config panel toggle (reserved for future use)

  const tracks = useMemo(() => tracksProp || buildDefaultTracks(data), [tracksProp, data]);

  const indexValues = data.index_values;
  // Compute scale: pixels per index unit
  const chartHeight = height - HEADER_HEIGHT - PADDING * 2;
  const totalHeight = chartHeight * zoom;

  // Depth ticks
  const depthTicks = useMemo(() => {
    if (indexValues.length === 0) return [];
    const min = indexValues[0];
    const max = indexValues[indexValues.length - 1];
    const range = max - min;
    // Determine tick interval
    let interval = 10;
    if (range > 1000) interval = 100;
    if (range > 5000) interval = 500;
    if (range > 10000) interval = 1000;

    const ticks: number[] = [];
    const start = Math.ceil(min / interval) * interval;
    for (let d = start; d <= max; d += interval) {
      ticks.push(d);
    }
    return ticks;
  }, [indexValues]);

  const depthToY = (depth: number): number => {
    if (indexValues.length < 2) return 0;
    const min = indexValues[0];
    const max = indexValues[indexValues.length - 1];
    return ((depth - min) / (max - min)) * totalHeight;
  };

  // Build SVG path for a curve
  const buildCurvePath = (mnemonic: string, trackX: number, trackW: number): { path: string; min: number; max: number } => {
    const curveData = data.curves[mnemonic];
    if (!curveData) return { path: '', min: 0, max: 1 };

    const values = curveData.values;
    // Compute min/max for scaling (ignore nulls)
    const validVals = values.filter((v): v is number => v !== null && !isNaN(v));
    if (validVals.length === 0) return { path: '', min: 0, max: 1 };

    const min = Math.min(...validVals);
    const max = Math.max(...validVals);
    const range = max - min || 1;

    let pathParts: string[] = [];
    let inPath = false;

    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (v === null || isNaN(v)) {
        inPath = false;
        continue;
      }
      const x = trackX + ((v - min) / range) * trackW;
      const y = depthToY(indexValues[i]);
      if (!inPath) {
        pathParts.push(`M${x.toFixed(1)},${y.toFixed(1)}`);
        inPath = true;
      } else {
        pathParts.push(`L${x.toFixed(1)},${y.toFixed(1)}`);
      }
    }

    return { path: pathParts.join(' '), min, max };
  };

  const totalWidth = RULER_WIDTH + tracks.length * TRACK_WIDTH;

  // Scroll to current depth
  useEffect(() => {
    if (currentDepth !== undefined && containerRef.current && indexValues.length > 0) {
      const y = depthToY(currentDepth);
      const container = containerRef.current;
      const visibleHeight = container.clientHeight;
      if (y < scrollTop || y > scrollTop + visibleHeight - 40) {
        container.scrollTop = Math.max(0, y - visibleHeight / 2);
      }
    }
  }, [currentDepth]);

  if (indexValues.length === 0) {
    return (
      <Card className="border-border bg-card/30">
        <CardContent className="pt-6 text-center text-muted-foreground text-sm font-mono">
          No curve data loaded
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="border-border bg-card/30 overflow-hidden">
      <CardHeader className="pb-2 border-b border-border/50">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-mono flex items-center gap-2">
            <Layers className="w-4 h-4 text-chart-3" />
            LOG TRACK VIEWER
          </CardTitle>
          <div className="flex items-center gap-1">
            <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={() => setZoom(z => Math.max(0.5, z - 0.25))}>
              <ZoomOut className="w-3.5 h-3.5" />
            </Button>
            <span className="text-[10px] font-mono text-muted-foreground w-10 text-center">{(zoom * 100).toFixed(0)}%</span>
            <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={() => setZoom(z => Math.min(4, z + 0.25))}>
              <ZoomIn className="w-3.5 h-3.5" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        {/* Track headers */}
        <div className="flex border-b border-border/50 bg-card/60" style={{ paddingLeft: RULER_WIDTH }}>
          {tracks.map((track) => (
            <div
              key={track.id}
              className="border-r border-border/30 px-2 py-1 text-center"
              style={{ width: TRACK_WIDTH }}
            >
              <div className="text-[10px] font-mono font-medium truncate" style={{ color: track.color }}>
                {track.label}
              </div>
              <div className="text-[9px] text-muted-foreground truncate">
                {track.curves.join(', ')}
              </div>
            </div>
          ))}
        </div>

        {/* Scrollable log area */}
        <div
          ref={containerRef}
          className="overflow-auto"
          style={{ height: chartHeight }}
          onScroll={e => setScrollTop((e.target as HTMLDivElement).scrollTop)}
        >
          <svg width={totalWidth} height={totalHeight} className="block">
            {/* Background */}
            <rect width={totalWidth} height={totalHeight} fill="var(--card)" opacity={0.3} />

            {/* Depth ruler */}
            {depthTicks.map(d => {
              const y = depthToY(d);
              return (
                <g key={d}>
                  <line x1={0} y1={y} x2={totalWidth} y2={y} stroke="var(--border)" strokeWidth={0.5} opacity={0.3} />
                  <text x={RULER_WIDTH - 4} y={y + 3} textAnchor="end" fontSize={9} fill="var(--muted-foreground)" fontFamily="monospace">
                    {d.toFixed(0)}
                  </text>
                </g>
              );
            })}

            {/* Track separators */}
            {tracks.map((_, i) => {
              const x = RULER_WIDTH + (i + 1) * TRACK_WIDTH;
              return (
                <line key={`sep-${i}`} x1={x} y1={0} x2={x} y2={totalHeight} stroke="var(--border)" strokeWidth={0.5} opacity={0.3} />
              );
            })}

            {/* Curve paths */}
            {tracks.map((track, trackIdx) => {
              const trackX = RULER_WIDTH + trackIdx * TRACK_WIDTH + 4;
              const trackW = TRACK_WIDTH - 8;
              return track.curves.map((mnem, curveIdx) => {
                const { path, min, max } = buildCurvePath(mnem, trackX, trackW);
                if (!path) return null;
                return (
                  <g key={`${track.id}-${mnem}`}>
                    <path
                      d={path}
                      fill="none"
                      stroke={CURVE_PALETTE[curveIdx % CURVE_PALETTE.length]}
                      strokeWidth={1.2}
                      opacity={0.9}
                    />
                    {/* Scale labels */}
                    <text x={trackX} y={12} fontSize={8} fill="var(--muted-foreground)" fontFamily="monospace">
                      {min.toFixed(1)}
                    </text>
                    <text x={trackX + trackW} y={12} textAnchor="end" fontSize={8} fill="var(--muted-foreground)" fontFamily="monospace">
                      {max.toFixed(1)}
                    </text>
                  </g>
                );
              });
            })}

            {/* Current depth indicator */}
            {currentDepth !== undefined && (
              <line
                x1={0}
                y1={depthToY(currentDepth)}
                x2={totalWidth}
                y2={depthToY(currentDepth)}
                stroke="#ef4444"
                strokeWidth={1.5}
                strokeDasharray="4 2"
              />
            )}
          </svg>
        </div>
      </CardContent>
    </Card>
  );
}
