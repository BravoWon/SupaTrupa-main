import { useMemo } from 'react';
import type { AdvisoryResponse } from '@/types';

// Parameter color mapping (consistent with AdvisoryPanel)
const PARAM_COLORS: Record<string, string> = {
  wob: '#2dd4bf',
  rpm: '#60a5fa',
  rop: '#a78bfa',
  torque: '#f59e0b',
  spp: '#f472b6',
};

const RISK_COLORS: Record<string, string> = {
  low: '#22c55e',
  medium: '#eab308',
  high: '#f97316',
  critical: '#ef4444',
};

interface Props {
  advisory: AdvisoryResponse | null;
}

const W = 600;
const H = 400;
const PAD = { top: 30, right: 30, bottom: 30, left: 60 };

export function GeodesicNavigator({ advisory }: Props) {
  if (!advisory || advisory.parameter_trajectory.length < 2) {
    return (
      <div className="h-full flex items-center justify-center text-muted-foreground font-mono text-xs">
        <div className="text-center space-y-1">
          <div className="text-sm">GEODESIC NAVIGATOR</div>
          <div className="text-[10px] opacity-60">
            Awaiting advisory computation...
          </div>
        </div>
      </div>
    );
  }

  const trajectory = advisory.parameter_trajectory;
  const params = Object.keys(trajectory[0]).filter((k) => k in PARAM_COLORS);

  // Compute scales for each parameter
  const scales = useMemo(() => {
    const result: Record<string, { min: number; max: number }> = {};
    for (const p of params) {
      const vals = trajectory.map((t) => t[p] ?? 0);
      const min = Math.min(...vals);
      const max = Math.max(...vals);
      const pad = Math.max((max - min) * 0.1, 0.1);
      result[p] = { min: min - pad, max: max + pad };
    }
    return result;
  }, [trajectory, params]);

  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;

  // X maps to trajectory step index
  const xScale = (i: number) => PAD.left + (i / (trajectory.length - 1)) * innerW;

  // Y maps parameter value to SVG y
  const yScale = (param: string, value: number) => {
    const { min, max } = scales[param] ?? { min: 0, max: 1 };
    const norm = (value - min) / (max - min || 1);
    return PAD.top + innerH * (1 - norm);
  };

  // Build paths for each parameter
  const paths = useMemo(() => {
    return params.map((param) => {
      const points = trajectory.map((t, i) => {
        const x = xScale(i);
        const y = yScale(param, t[param] ?? 0);
        return `${x},${y}`;
      });
      return {
        param,
        d: `M${points.join('L')}`,
        color: PARAM_COLORS[param] ?? '#94a3b8',
      };
    });
  }, [trajectory, params, scales]);

  // Step markers (where each step's change begins)
  const stepMarkers = useMemo(() => {
    if (!advisory.steps.length) return [];
    const markers: { x: number; param: string; priority: number }[] = [];
    let accum = 0;
    const stepsPerParam = Math.max(1, Math.floor(trajectory.length / advisory.steps.length));
    for (const step of advisory.steps) {
      accum += stepsPerParam;
      markers.push({
        x: xScale(Math.min(accum, trajectory.length - 1)),
        param: step.parameter,
        priority: step.priority,
      });
    }
    return markers;
  }, [advisory.steps, trajectory]);

  const riskColor = RISK_COLORS[advisory.risk_level] ?? '#eab308';

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Header */}
      <div className="shrink-0 border-b border-border/50 px-2 py-1.5">
        <div className="flex items-center justify-between">
          <span className="text-[9px] font-mono text-muted-foreground uppercase tracking-wider">
            Geodesic Navigator
          </span>
          <div className="flex items-center gap-2">
            {/* Path efficiency indicator */}
            <span className="text-[9px] font-mono text-muted-foreground">
              eff: {advisory.path_efficiency.toFixed(2)}x
            </span>
            <div
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: riskColor }}
            />
          </div>
        </div>
      </div>

      {/* SVG chart */}
      <div className="flex-1 min-h-0 relative">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="w-full h-full"
          preserveAspectRatio="xMidYMid meet"
        >
          {/* Grid lines */}
          {Array.from({ length: 6 }, (_, i) => {
            const y = PAD.top + (innerH * i) / 5;
            return (
              <line
                key={`grid-${i}`}
                x1={PAD.left}
                y1={y}
                x2={W - PAD.right}
                y2={y}
                stroke="currentColor"
                strokeOpacity={0.06}
              />
            );
          })}

          {/* X-axis step labels */}
          {trajectory.length > 1 &&
            [0, Math.floor(trajectory.length / 2), trajectory.length - 1].map((idx) => (
              <text
                key={`x-${idx}`}
                x={xScale(idx)}
                y={H - 8}
                textAnchor="middle"
                className="text-[8px] fill-muted-foreground font-mono"
              >
                {idx === 0 ? 'START' : idx === trajectory.length - 1 ? 'TARGET' : `S${Math.floor(idx)}`}
              </text>
            ))}

          {/* Step divider lines */}
          {stepMarkers.map((m, i) => (
            <g key={`step-${i}`}>
              <line
                x1={m.x}
                y1={PAD.top}
                x2={m.x}
                y2={PAD.top + innerH}
                stroke={PARAM_COLORS[m.param] ?? '#94a3b8'}
                strokeOpacity={0.25}
                strokeDasharray="3,3"
              />
              <text
                x={m.x}
                y={PAD.top - 8}
                textAnchor="middle"
                className="text-[8px] font-mono"
                fill={PARAM_COLORS[m.param] ?? '#94a3b8'}
              >
                {m.priority}
              </text>
            </g>
          ))}

          {/* Parameter trajectory lines */}
          {paths.map(({ param, d, color }) => (
            <g key={param}>
              {/* Shadow */}
              <path d={d} fill="none" stroke={color} strokeWidth={3} strokeOpacity={0.15} />
              {/* Main line */}
              <path d={d} fill="none" stroke={color} strokeWidth={1.5} strokeOpacity={0.8} />
              {/* Start dot */}
              <circle
                cx={xScale(0)}
                cy={yScale(param, trajectory[0][param] ?? 0)}
                r={3}
                fill={color}
                fillOpacity={0.6}
              />
              {/* End dot */}
              <circle
                cx={xScale(trajectory.length - 1)}
                cy={yScale(param, trajectory[trajectory.length - 1][param] ?? 0)}
                r={3}
                fill={color}
                fillOpacity={0.9}
                stroke={color}
                strokeWidth={1}
              />
            </g>
          ))}

          {/* Y-axis labels (parameter names) */}
          {params.map((param) => {
            const lastVal = trajectory[trajectory.length - 1][param] ?? 0;
            const y = yScale(param, lastVal);
            return (
              <text
                key={`label-${param}`}
                x={W - PAD.right + 4}
                y={y}
                dominantBaseline="middle"
                className="text-[8px] font-mono"
                fill={PARAM_COLORS[param] ?? '#94a3b8'}
              >
                {param.toUpperCase()}
              </text>
            );
          })}
        </svg>
      </div>

      {/* Legend bar */}
      <div className="shrink-0 border-t border-border/50 px-2 py-1 flex items-center gap-3 flex-wrap">
        {params.map((param) => (
          <div key={param} className="flex items-center gap-1">
            <div
              className="w-2 h-2 rounded-sm"
              style={{ backgroundColor: PARAM_COLORS[param] ?? '#94a3b8' }}
            />
            <span className="text-[8px] font-mono text-muted-foreground">
              {param.toUpperCase()}
            </span>
          </div>
        ))}
        <div className="flex-1" />
        <span className="text-[8px] font-mono text-muted-foreground">
          Length: {advisory.geodesic_length.toFixed(1)} | Euclidean: {advisory.euclidean_length.toFixed(1)}
        </span>
      </div>
    </div>
  );
}
