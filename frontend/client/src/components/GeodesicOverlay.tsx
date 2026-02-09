import { useMemo } from 'react';
import { cn } from '@/lib/utils';
import type { GeodesicResponse } from '@/types';

interface GeodesicOverlayProps {
  geodesic: GeodesicResponse | null;
  tRange: [number, number];
  dRange: [number, number];
  className?: string;
}

/**
 * SVG overlay showing geodesic paths on the drilling manifold.
 * Displays the shortest path between two drilling states
 * through ROP-warped geometry with curvature annotations.
 */
export function GeodesicOverlay({
  geodesic,
  tRange,
  dRange,
  className,
}: GeodesicOverlayProps) {
  const svgW = 400;
  const svgH = 300;
  const padX = 48;
  const padY = 24;
  const plotW = svgW - padX * 2;
  const plotH = svgH - padY * 2;

  const { pathD, straightD, startPt, endPt, midpoints } = useMemo(() => {
    if (!geodesic || geodesic.path.length < 2) {
      return { pathD: '', straightD: '', startPt: null, endPt: null, midpoints: [] as { x: number; y: number; rop: number }[] };
    }

    const tMin = tRange[0];
    const tMax = tRange[1];
    const dMin = dRange[0];
    const dMax = dRange[1];
    const tSpan = tMax - tMin || 1;
    const dSpan = dMax - dMin || 1;

    const xScale = (t: number) => padX + ((t - tMin) / tSpan) * plotW;
    const yScale = (d: number) => padY + plotH - ((d - dMin) / dSpan) * plotH;

    const pts = geodesic.path.map(([t, d]) => ({ x: xScale(t), y: yScale(d) }));

    const pathD = pts
      .map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(1)},${p.y.toFixed(1)}`)
      .join(' ');

    // Straight Euclidean line for comparison
    const straightD = `M${pts[0].x.toFixed(1)},${pts[0].y.toFixed(1)} L${pts[pts.length - 1].x.toFixed(1)},${pts[pts.length - 1].y.toFixed(1)}`;

    // Sample midpoints for ROP labels
    const step = Math.max(1, Math.floor(geodesic.path.length / 4));
    const midpoints: { x: number; y: number; rop: number }[] = [];
    for (let i = step; i < geodesic.path.length - step; i += step) {
      const [t, d] = geodesic.path[i];
      midpoints.push({
        x: xScale(t),
        y: yScale(d),
        rop: 0, // We don't have per-point ROP from the response, so estimate
      });
    }

    return {
      pathD,
      straightD,
      startPt: pts[0],
      endPt: pts[pts.length - 1],
      midpoints,
    };
  }, [geodesic, tRange, dRange]);

  if (!geodesic || geodesic.path.length < 2) {
    return (
      <div className={cn('flex items-center justify-center text-muted-foreground text-[10px] font-mono h-full', className)}>
        <div className="text-center">
          <div className="text-[10px] uppercase tracking-wider mb-1">Optimal Path</div>
          <div className="text-[9px]">Select start/end points to compute</div>
        </div>
      </div>
    );
  }

  return (
    <div className={cn('w-full h-full flex flex-col', className)}>
      <div className="text-[9px] font-mono text-muted-foreground uppercase tracking-wider px-2 py-1 shrink-0 flex items-center justify-between">
        <span>Optimal Path</span>
        <span className="text-[8px] text-foreground/70">
          Length: {geodesic.total_length.toFixed(2)}
        </span>
      </div>
      <svg
        viewBox={`0 0 ${svgW} ${svgH}`}
        className="w-full flex-1"
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Grid */}
        {[0, 0.25, 0.5, 0.75, 1].map(frac => {
          const y = padY + plotH * (1 - frac);
          const x = padX + plotW * frac;
          return (
            <g key={frac}>
              <line x1={padX} y1={y} x2={svgW - padX} y2={y}
                stroke="hsl(var(--border))" strokeWidth={0.3} strokeDasharray="2,4" />
              <line x1={x} y1={padY} x2={x} y2={padY + plotH}
                stroke="hsl(var(--border))" strokeWidth={0.3} strokeDasharray="2,4" />
            </g>
          );
        })}

        {/* Axis labels */}
        {[0, 0.5, 1].map(frac => {
          const tVal = tRange[0] + (tRange[1] - tRange[0]) * frac;
          const dVal = dRange[0] + (dRange[1] - dRange[0]) * frac;
          const x = padX + plotW * frac;
          const y = padY + plotH * (1 - frac);
          return (
            <g key={`label-${frac}`}>
              <text x={x} y={padY + plotH + 12} textAnchor="middle"
                className="fill-muted-foreground" fontSize={7} fontFamily="monospace">
                {tVal.toFixed(0)}
              </text>
              <text x={padX - 4} y={y + 3} textAnchor="end"
                className="fill-muted-foreground" fontSize={7} fontFamily="monospace">
                {dVal.toFixed(0)}
              </text>
            </g>
          );
        })}

        {/* Straight Euclidean line (comparison) */}
        <path d={straightD} fill="none" stroke="hsl(var(--muted-foreground))"
          strokeWidth={1} strokeDasharray="4,4" opacity={0.3} />

        {/* Geodesic path (main) */}
        <path d={pathD} fill="none" stroke="#14b8a6" strokeWidth={2} />

        {/* Glow effect */}
        <path d={pathD} fill="none" stroke="#14b8a6" strokeWidth={6} opacity={0.15} />

        {/* Start point */}
        {startPt && (
          <g>
            <circle cx={startPt.x} cy={startPt.y} r={5} fill="#14b8a6" stroke="hsl(var(--background))" strokeWidth={1.5} />
            <text x={startPt.x + 8} y={startPt.y - 6}
              className="fill-foreground" fontSize={7} fontFamily="monospace">
              START (R={geodesic.start_curvature.toFixed(3)})
            </text>
            <text x={startPt.x + 8} y={startPt.y + 4}
              className="fill-muted-foreground" fontSize={6} fontFamily="monospace">
              ROP={geodesic.start_rop.toFixed(1)} ft/hr
            </text>
          </g>
        )}

        {/* End point */}
        {endPt && (
          <g>
            <circle cx={endPt.x} cy={endPt.y} r={5} fill="#f59e0b" stroke="hsl(var(--background))" strokeWidth={1.5} />
            <text x={endPt.x + 8} y={endPt.y - 6}
              className="fill-foreground" fontSize={7} fontFamily="monospace">
              END (R={geodesic.end_curvature.toFixed(3)})
            </text>
            <text x={endPt.x + 8} y={endPt.y + 4}
              className="fill-muted-foreground" fontSize={6} fontFamily="monospace">
              ROP={geodesic.end_rop.toFixed(1)} ft/hr
            </text>
          </g>
        )}

        {/* Path direction arrows */}
        {midpoints.map((mp, i) => (
          <circle key={`mid-${i}`} cx={mp.x} cy={mp.y} r={2}
            fill="#14b8a6" opacity={0.6} />
        ))}

        {/* Legend */}
        <g transform={`translate(${svgW - padX - 90}, ${padY + 4})`}>
          <line x1={0} y1={4} x2={14} y2={4} stroke="#14b8a6" strokeWidth={2} />
          <text x={18} y={7} className="fill-foreground" fontSize={7} fontFamily="monospace">
            Optimal
          </text>
          <line x1={0} y1={14} x2={14} y2={14} stroke="hsl(var(--muted-foreground))"
            strokeWidth={1} strokeDasharray="3,3" opacity={0.5} />
          <text x={18} y={17} className="fill-muted-foreground" fontSize={7} fontFamily="monospace">
            Straight-line
          </text>
        </g>

        {/* Axis titles */}
        <text x={svgW / 2} y={svgH - 4} textAnchor="middle"
          className="fill-muted-foreground" fontSize={8} fontFamily="monospace">
          Time (s)
        </text>
        <text x={10} y={padY + plotH / 2} textAnchor="middle"
          className="fill-muted-foreground" fontSize={8} fontFamily="monospace"
          transform={`rotate(-90, 10, ${padY + plotH / 2})`}>
          Depth (ft)
        </text>
      </svg>
    </div>
  );
}
