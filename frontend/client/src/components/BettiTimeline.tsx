import { useMemo } from 'react';
import { cn } from '@/lib/utils';
import type { WindowedSignature } from '@/types';

interface BettiTimelineProps {
  windows: WindowedSignature[];
  className?: string;
}

/**
 * SVG sparkline showing Betti_0 (teal) and Betti_1 (amber) over sliding windows.
 * Vertical dashed lines mark windows where entropy spikes (potential regime changes).
 */
export function BettiTimeline({ windows, className }: BettiTimelineProps) {
  const { pathH0, pathH1, changeMarkers, maxBetti, svgW, svgH, padX, padY } = useMemo(() => {
    const svgW = 400;
    const svgH = 120;
    const padX = 32;
    const padY = 16;

    if (windows.length < 2) {
      return { pathH0: '', pathH1: '', changeMarkers: [] as number[], maxBetti: 1, svgW, svgH, padX, padY };
    }

    const plotW = svgW - padX * 2;
    const plotH = svgH - padY * 2;

    const b0 = windows.map(w => w.betti_0);
    const b1 = windows.map(w => w.betti_1);
    const maxBetti = Math.max(1, ...b0, ...b1);

    const xScale = (i: number) => padX + (i / (windows.length - 1)) * plotW;
    const yScale = (v: number) => padY + plotH - (v / maxBetti) * plotH;

    const toPath = (vals: number[]) =>
      vals.map((v, i) => `${i === 0 ? 'M' : 'L'}${xScale(i).toFixed(1)},${yScale(v).toFixed(1)}`).join(' ');

    // Detect entropy spikes as potential regime change markers
    const changeMarkers: number[] = [];
    for (let i = 1; i < windows.length; i++) {
      const dH0 = Math.abs(windows[i].entropy_h0 - windows[i - 1].entropy_h0);
      const dH1 = Math.abs(windows[i].entropy_h1 - windows[i - 1].entropy_h1);
      if (dH0 > 0.5 || dH1 > 0.5) {
        changeMarkers.push(i);
      }
    }

    return {
      pathH0: toPath(b0),
      pathH1: toPath(b1),
      changeMarkers,
      maxBetti,
      svgW,
      svgH,
      padX,
      padY,
    };
  }, [windows]);

  if (windows.length < 2) {
    return (
      <div className={cn('flex items-center justify-center text-muted-foreground text-[10px] font-mono h-full', className)}>
        Awaiting windowed data...
      </div>
    );
  }

  const plotW = svgW - padX * 2;
  const plotH = svgH - padY * 2;
  const xScale = (i: number) => padX + (i / (windows.length - 1)) * plotW;

  return (
    <div className={cn('w-full h-full flex flex-col', className)}>
      <div className="text-[9px] font-mono text-muted-foreground uppercase tracking-wider px-2 py-1 shrink-0">
        Zone / Coupling History ({windows.length} windows)
      </div>
      <svg
        viewBox={`0 0 ${svgW} ${svgH}`}
        className="w-full flex-1"
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map(frac => {
          const y = padY + plotH - frac * plotH;
          return (
            <g key={frac}>
              <line
                x1={padX} y1={y} x2={svgW - padX} y2={y}
                stroke="hsl(var(--border))" strokeWidth={0.5} strokeDasharray="2,3"
              />
              <text x={padX - 4} y={y + 3} textAnchor="end"
                className="fill-muted-foreground" fontSize={7} fontFamily="monospace">
                {Math.round(frac * maxBetti)}
              </text>
            </g>
          );
        })}

        {/* Regime change markers */}
        {changeMarkers.map(i => (
          <line
            key={`cm-${i}`}
            x1={xScale(i)} y1={padY} x2={xScale(i)} y2={svgH - padY}
            stroke="hsl(var(--destructive))" strokeWidth={1} strokeDasharray="3,3" opacity={0.6}
          />
        ))}

        {/* Betti_0 line (teal) */}
        <path d={pathH0} fill="none" stroke="#14b8a6" strokeWidth={1.5} />
        {/* Betti_1 line (amber) */}
        <path d={pathH1} fill="none" stroke="#f59e0b" strokeWidth={1.5} />

        {/* Area fills (subtle) */}
        {pathH0 && (
          <path
            d={`${pathH0} L${(svgW - padX).toFixed(1)},${(svgH - padY).toFixed(1)} L${padX},${(svgH - padY).toFixed(1)} Z`}
            fill="#14b8a6" opacity={0.08}
          />
        )}
        {pathH1 && (
          <path
            d={`${pathH1} L${(svgW - padX).toFixed(1)},${(svgH - padY).toFixed(1)} L${padX},${(svgH - padY).toFixed(1)} Z`}
            fill="#f59e0b" opacity={0.08}
          />
        )}

        {/* Legend */}
        <g transform={`translate(${svgW - padX - 80}, ${padY})`}>
          <line x1={0} y1={4} x2={12} y2={4} stroke="#14b8a6" strokeWidth={1.5} />
          <text x={16} y={7} className="fill-foreground" fontSize={7} fontFamily="monospace">
            Zones
          </text>
          <line x1={0} y1={14} x2={12} y2={14} stroke="#f59e0b" strokeWidth={1.5} />
          <text x={16} y={17} className="fill-foreground" fontSize={7} fontFamily="monospace">
            Couplings
          </text>
        </g>

        {/* X-axis label */}
        <text x={svgW / 2} y={svgH - 2} textAnchor="middle"
          className="fill-muted-foreground" fontSize={7} fontFamily="monospace">
          Window Index
        </text>
      </svg>
    </div>
  );
}
