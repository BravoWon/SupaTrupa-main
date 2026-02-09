import { useMemo } from 'react';
import { cn } from '@/lib/utils';
import type { ForecastResponse, WindowedSignature } from '@/types';

interface TopologyForecastProps {
  forecast: ForecastResponse | null;
  history: WindowedSignature[];
  className?: string;
}


/**
 * SVG chart showing historical Betti trajectory with forecast extension
 * and confidence bands. History is solid, forecast is dashed with shaded bands.
 */
export function TopologyForecast({ forecast, history, className }: TopologyForecastProps) {
  const svgW = 500;
  const svgH = 260;
  const padL = 40;
  const padR = 15;
  const padT = 25;
  const padB = 35;
  const plotW = svgW - padL - padR;
  const plotH = svgH - padT - padB;

  const { histB0, histB1, foreB0, foreB1, upperB0, upperB1, xScale, totalLen, histLen } = useMemo(() => {
    const empty = { histB0: '', histB1: '', foreB0: '', foreB1: '', upperB0: '', lowerB0: '', upperB1: '', lowerB1: '', xScale: (_: number) => 0, yScale: (_: number) => 0, totalLen: 0, histLen: 0 };
    if (history.length === 0 && !forecast) return empty;

    const histLen = history.length;
    const foreLen = forecast?.forecast.length ?? 0;
    const totalLen = histLen + foreLen;
    if (totalLen === 0) return empty;

    // Gather all values for y-range
    let yMin = Infinity;
    let yMax = -Infinity;
    for (const w of history) {
      yMin = Math.min(yMin, w.betti_0, w.betti_1);
      yMax = Math.max(yMax, w.betti_0, w.betti_1);
    }
    if (forecast) {
      for (const fp of forecast.forecast) {
        yMin = Math.min(yMin, fp.betti_0, fp.betti_1);
        yMax = Math.max(yMax, fp.betti_0, fp.betti_1);
        const ub0 = fp.confidence_upper?.betti_0 ?? fp.betti_0;
        const ub1 = fp.confidence_upper?.betti_1 ?? fp.betti_1;
        yMax = Math.max(yMax, ub0, ub1);
      }
    }
    if (yMin === yMax) { yMin -= 1; yMax += 1; }
    const yPad = (yMax - yMin) * 0.15;
    yMin -= yPad;
    yMax += yPad;

    const xScale = (i: number) => padL + (i / (totalLen - 1 || 1)) * plotW;
    const yScale = (v: number) => padT + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

    // History polylines
    const toLine = (vals: number[], offset: number) =>
      vals.map((v, i) => `${xScale(i + offset).toFixed(1)},${yScale(v).toFixed(1)}`).join(' ');

    const histB0 = toLine(history.map(w => w.betti_0), 0);
    const histB1 = toLine(history.map(w => w.betti_1), 0);

    // Forecast polylines
    let foreB0 = '';
    let foreB1 = '';
    let upperB0 = '';
    let lowerB0 = '';
    let upperB1 = '';
    let lowerB1 = '';

    if (forecast && forecast.forecast.length > 0) {
      // Connect from last history point
      const startIdx = histLen - 1;
      const forecastPts = forecast.forecast;

      const b0Vals = [history.length > 0 ? history[history.length - 1].betti_0 : forecastPts[0].betti_0, ...forecastPts.map(f => f.betti_0)];
      const b1Vals = [history.length > 0 ? history[history.length - 1].betti_1 : forecastPts[0].betti_1, ...forecastPts.map(f => f.betti_1)];

      foreB0 = b0Vals.map((v, i) => `${xScale(startIdx + i).toFixed(1)},${yScale(v).toFixed(1)}`).join(' ');
      foreB1 = b1Vals.map((v, i) => `${xScale(startIdx + i).toFixed(1)},${yScale(v).toFixed(1)}`).join(' ');

      // Confidence band paths (area)
      const ub0 = forecastPts.map(f => f.confidence_upper?.betti_0 ?? f.betti_0);
      const lb0 = forecastPts.map(f => f.confidence_lower?.betti_0 ?? f.betti_0);
      const ub1 = forecastPts.map(f => f.confidence_upper?.betti_1 ?? f.betti_1);
      const lb1 = forecastPts.map(f => f.confidence_lower?.betti_1 ?? f.betti_1);

      const bandPath = (upper: number[], lower: number[]) => {
        const top = upper.map((v, i) => `${xScale(histLen + i).toFixed(1)},${yScale(v).toFixed(1)}`);
        const bottom = [...lower].reverse().map((v, i) => `${xScale(histLen + lower.length - 1 - i).toFixed(1)},${yScale(v).toFixed(1)}`);
        return `M ${top.join(' L ')} L ${bottom.join(' L ')} Z`;
      };

      upperB0 = bandPath(ub0, lb0);
      lowerB0 = ''; // unused, band is a closed path
      upperB1 = bandPath(ub1, lb1);
      lowerB1 = '';
    }

    return { histB0, histB1, foreB0, foreB1, upperB0, lowerB0, upperB1, lowerB1, xScale, yScale, totalLen, histLen };
  }, [history, forecast]);

  if (!forecast && history.length === 0) {
    return (
      <div className={cn('flex items-center justify-center text-muted-foreground text-[10px] font-mono h-full', className)}>
        Awaiting forecast data...
      </div>
    );
  }

  const trendColor = forecast?.trend_direction === 'stable' ? '#22c55e' :
    forecast?.trend_direction === 'converging' ? '#eab308' : '#ef4444';

  return (
    <div className={cn('w-full h-full flex flex-col', className)}>
      {/* Header */}
      <div className="text-[9px] font-mono text-muted-foreground uppercase tracking-wider px-2 py-1 shrink-0 flex justify-between">
        <span>Topology Forecast</span>
        <div className="flex gap-3">
          {forecast && (
            <>
              <span style={{ color: trendColor }}>
                {forecast.trend_direction}
              </span>
              <span>
                stability={forecast.stability_index.toFixed(2)}
              </span>
            </>
          )}
        </div>
      </div>

      {/* Chart */}
      <svg viewBox={`0 0 ${svgW} ${svgH}`} className="w-full flex-1" preserveAspectRatio="xMidYMid meet">
        {/* Grid lines */}
        {[0.25, 0.5, 0.75].map(frac => {
          const y = padT + plotH * (1 - frac);
          return (
            <line key={frac} x1={padL} y1={y} x2={svgW - padR} y2={y}
              stroke="hsl(var(--border))" strokeWidth={0.3} strokeDasharray="3,5" />
          );
        })}

        {/* Forecast/history divider */}
        {histLen > 0 && totalLen > histLen && (
          <line
            x1={xScale(histLen - 1)} y1={padT}
            x2={xScale(histLen - 1)} y2={padT + plotH}
            stroke="hsl(var(--border))" strokeWidth={0.5} strokeDasharray="4,4"
          />
        )}
        {histLen > 0 && totalLen > histLen && (
          <text x={xScale(histLen - 1) + 3} y={padT + 8}
            fill="hsl(var(--muted-foreground))" fontSize={7} fontFamily="monospace">
            forecast
          </text>
        )}

        {/* Confidence bands (shaded) */}
        {upperB0 && (
          <path d={upperB0} fill="#14b8a6" fillOpacity={0.08} stroke="none" />
        )}
        {upperB1 && (
          <path d={upperB1} fill="#f59e0b" fillOpacity={0.08} stroke="none" />
        )}

        {/* History lines (solid) */}
        {histB0 && <polyline points={histB0} fill="none" stroke="#14b8a6" strokeWidth={1.5} />}
        {histB1 && <polyline points={histB1} fill="none" stroke="#f59e0b" strokeWidth={1.5} />}

        {/* Forecast lines (dashed) */}
        {foreB0 && <polyline points={foreB0} fill="none" stroke="#14b8a6" strokeWidth={1.5} strokeDasharray="5,3" />}
        {foreB1 && <polyline points={foreB1} fill="none" stroke="#f59e0b" strokeWidth={1.5} strokeDasharray="5,3" />}

        {/* X axis label */}
        <text x={padL + plotW / 2} y={svgH - 5} textAnchor="middle"
          fill="hsl(var(--muted-foreground))" fontSize={8} fontFamily="monospace">
          Window Index
        </text>

        {/* Y axis label */}
        <text x={8} y={padT + plotH / 2} textAnchor="middle"
          fill="hsl(var(--muted-foreground))" fontSize={8} fontFamily="monospace"
          transform={`rotate(-90, 8, ${padT + plotH / 2})`}>
          Zone / Coupling Count
        </text>

        {/* Legend */}
        <g transform={`translate(${padL + 5}, ${svgH - 22})`}>
          <line x1={0} y1={0} x2={14} y2={0} stroke="#14b8a6" strokeWidth={1.5} />
          <text x={18} y={3} fill="hsl(var(--foreground))" fontSize={7} fontFamily="monospace">Zones</text>
          <line x1={50} y1={0} x2={64} y2={0} stroke="#f59e0b" strokeWidth={1.5} />
          <text x={68} y={3} fill="hsl(var(--foreground))" fontSize={7} fontFamily="monospace">Couplings</text>
          <line x1={80} y1={0} x2={94} y2={0} stroke="hsl(var(--muted-foreground))" strokeWidth={1} strokeDasharray="4,3" />
          <text x={98} y={3} fill="hsl(var(--muted-foreground))" fontSize={7} fontFamily="monospace">forecast</text>
        </g>
      </svg>

      {/* Velocity summary */}
      {forecast && (
        <div className="px-2 py-1 text-[8px] font-mono text-muted-foreground flex gap-4 shrink-0 border-t border-border/30">
          <span>Zone velocity: <span className="text-foreground/70">{forecast.velocity.betti_0?.toFixed(3) ?? '0'}</span></span>
          <span>Coupling velocity: <span className="text-foreground/70">{forecast.velocity.betti_1?.toFixed(3) ?? '0'}</span></span>
          <span>windows: {forecast.n_windows_used} used, {forecast.n_ahead} ahead</span>
        </div>
      )}
    </div>
  );
}
