import { useMemo } from 'react';
import { cn } from '@/lib/utils';
import type { AttractorAnalysis } from '@/types';

interface LyapunovIndicatorProps {
  analysis: AttractorAnalysis | null;
  className?: string;
}

const ATTRACTOR_LABELS: Record<string, string> = {
  fixed_point: 'Stable',
  limit_cycle: 'Cyclic',
  strange_attractor: 'Complex',
  quasi_periodic: 'Multi-Cycle',
  stochastic: 'Noisy',
  transient: 'Transitioning',
};

const ATTRACTOR_COLORS: Record<string, string> = {
  fixed_point: '#22c55e',
  limit_cycle: '#eab308',
  strange_attractor: '#ef4444',
  quasi_periodic: '#f97316',
  stochastic: '#8b5cf6',
  transient: '#6b7280',
};

/**
 * Gauge showing Lyapunov exponent with attractor classification.
 * Negative = stable (green), zero = periodic (yellow), positive = chaotic (red).
 * Also displays correlation dimension, recurrence rate, determinism, laminarity.
 */
export function LyapunovIndicator({ analysis, className }: LyapunovIndicatorProps) {
  // Map Lyapunov exponent to gauge angle (180 degrees arc)
  const gaugeAngle = useMemo(() => {
    if (!analysis) return 90;
    // Clamp to [-0.5, 0.5] range for display
    const clamped = Math.max(-0.5, Math.min(0.5, analysis.lyapunov_exponent));
    // Map [-0.5, 0.5] -> [0, 180] degrees
    return ((clamped + 0.5) / 1.0) * 180;
  }, [analysis]);

  const gaugeColor = useMemo(() => {
    if (!analysis) return '#6b7280';
    const le = analysis.lyapunov_exponent;
    if (le < -0.01) return '#22c55e';
    if (le <= 0.01) return '#eab308';
    if (le < 0.1) return '#f97316';
    return '#ef4444';
  }, [analysis]);

  if (!analysis) {
    return (
      <div className={cn('flex items-center justify-center text-muted-foreground text-[10px] font-mono h-full', className)}>
        Awaiting attractor analysis...
      </div>
    );
  }

  const attractorColor = ATTRACTOR_COLORS[analysis.attractor_type] || '#6b7280';
  const attractorLabel = ATTRACTOR_LABELS[analysis.attractor_type] || analysis.attractor_type;

  // SVG gauge parameters
  const svgW = 200;
  const svgH = 120;
  const gaugeCx = svgW / 2;
  const gaugeCy = 95;
  const gaugeR = 70;

  // Convert angle to SVG arc coordinates
  const needleAngle = (180 - gaugeAngle) * (Math.PI / 180);
  const needleX = gaugeCx + Math.cos(needleAngle) * (gaugeR - 8);
  const needleY = gaugeCy - Math.sin(needleAngle) * (gaugeR - 8);

  return (
    <div className={cn('w-full h-full flex flex-col', className)}>
      {/* Header */}
      <div className="text-[9px] font-mono text-muted-foreground uppercase tracking-wider px-2 py-1 shrink-0 flex justify-between">
        <span>Behavioral Analysis</span>
        <span style={{ color: attractorColor }}>{attractorLabel}</span>
      </div>

      {/* Lyapunov Gauge */}
      <div className="flex-shrink-0 flex justify-center">
        <svg viewBox={`0 0 ${svgW} ${svgH}`} className="w-[200px] h-[120px]">
          {/* Background arc */}
          <path
            d={`M ${gaugeCx - gaugeR} ${gaugeCy} A ${gaugeR} ${gaugeR} 0 0 1 ${gaugeCx + gaugeR} ${gaugeCy}`}
            fill="none" stroke="hsl(var(--border))" strokeWidth={8} strokeLinecap="round"
          />

          {/* Color gradient arc segments */}
          {/* Green (stable) */}
          <path
            d={describeArc(gaugeCx, gaugeCy, gaugeR, 180, 135)}
            fill="none" stroke="#22c55e" strokeWidth={8} strokeLinecap="round" opacity={0.4}
          />
          {/* Yellow (neutral) */}
          <path
            d={describeArc(gaugeCx, gaugeCy, gaugeR, 100, 80)}
            fill="none" stroke="#eab308" strokeWidth={8} strokeLinecap="round" opacity={0.4}
          />
          {/* Red (chaotic) */}
          <path
            d={describeArc(gaugeCx, gaugeCy, gaugeR, 45, 0)}
            fill="none" stroke="#ef4444" strokeWidth={8} strokeLinecap="round" opacity={0.4}
          />

          {/* Needle */}
          <line
            x1={gaugeCx} y1={gaugeCy}
            x2={needleX} y2={needleY}
            stroke={gaugeColor} strokeWidth={2} strokeLinecap="round"
          />
          <circle cx={gaugeCx} cy={gaugeCy} r={4} fill={gaugeColor} />

          {/* Labels */}
          <text x={gaugeCx - gaugeR - 2} y={gaugeCy + 12} textAnchor="middle"
            fill="#22c55e" fontSize={7} fontFamily="monospace">Stable</text>
          <text x={gaugeCx + gaugeR + 2} y={gaugeCy + 12} textAnchor="middle"
            fill="#ef4444" fontSize={7} fontFamily="monospace">Chaotic</text>

          {/* Value */}
          <text x={gaugeCx} y={gaugeCy - 10} textAnchor="middle"
            fill="hsl(var(--foreground))" fontSize={14} fontFamily="monospace" fontWeight="bold">
            {analysis.lyapunov_exponent >= 0 ? '+' : ''}{analysis.lyapunov_exponent.toFixed(4)}
          </text>
          <text x={gaugeCx} y={gaugeCy + 2} textAnchor="middle"
            fill="hsl(var(--muted-foreground))" fontSize={7} fontFamily="monospace">
            Predictability Index
          </text>
        </svg>
      </div>

      {/* Interpretation */}
      <div className="px-2 text-[9px] font-mono text-center shrink-0" style={{ color: gaugeColor }}>
        {analysis.lyapunov_interpretation}
      </div>

      {/* Metrics grid */}
      <div className="flex-1 min-h-0 overflow-y-auto px-2 py-1 space-y-1">
        <MetricRow label="Dynamic Complexity" value={analysis.correlation_dimension.toFixed(2)}
          bar={Math.min(1, analysis.correlation_dimension / 5)} color="#14b8a6" />
        <MetricRow label="Pattern Repetition" value={`${(analysis.recurrence_rate * 100).toFixed(1)}%`}
          bar={analysis.recurrence_rate} color="#06b6d4" />
        <MetricRow label="Behavioral Consistency" value={`${(analysis.determinism * 100).toFixed(1)}%`}
          bar={analysis.determinism} color="#8b5cf6" />
        <MetricRow label="Steady-State Tendency" value={`${(analysis.laminarity * 100).toFixed(1)}%`}
          bar={analysis.laminarity} color="#f59e0b" />
        <MetricRow label="State Persistence" value={analysis.trapping_time.toFixed(1)}
          bar={Math.min(1, analysis.trapping_time / 10)} color="#ec4899" />
      </div>
    </div>
  );
}

function MetricRow({ label, value, bar, color }: {
  label: string; value: string; bar: number; color: string;
}) {
  return (
    <div className="flex items-center gap-1 text-[9px] font-mono">
      <span className="text-muted-foreground w-24 shrink-0">{label}</span>
      <div className="flex-1 h-2 bg-muted/20 rounded-sm overflow-hidden">
        <div
          className="h-full rounded-sm transition-all duration-300"
          style={{ width: `${Math.min(100, bar * 100)}%`, backgroundColor: color, opacity: 0.6 }}
        />
      </div>
      <span className="w-12 text-right text-foreground/70">{value}</span>
    </div>
  );
}

/** Compute SVG arc path between two angles (degrees, measured from right going counter-clockwise). */
function describeArc(cx: number, cy: number, r: number, startAngle: number, endAngle: number): string {
  const startRad = (startAngle * Math.PI) / 180;
  const endRad = (endAngle * Math.PI) / 180;
  const x1 = cx + Math.cos(Math.PI - startRad) * r;
  const y1 = cy - Math.sin(Math.PI - startRad) * r;
  const x2 = cx + Math.cos(Math.PI - endRad) * r;
  const y2 = cy - Math.sin(Math.PI - endRad) * r;
  const largeArc = Math.abs(startAngle - endAngle) > 180 ? 1 : 0;
  return `M ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 0 ${x2} ${y2}`;
}
