import { useMemo } from 'react';
import { cn } from '@/lib/utils';
import type { TransitionProbResponse } from '@/types';

interface TransitionRadarProps {
  transition: TransitionProbResponse | null;
  className?: string;
}

const RISK_COLORS: Record<string, string> = {
  low: '#22c55e',
  medium: '#eab308',
  high: '#f97316',
  critical: '#ef4444',
};

const REGIME_SHORT: Record<string, string> = {
  NORMAL: 'NRM',
  OPTIMAL: 'OPT',
  DARCY_FLOW: 'DRC',
  NON_DARCY_FLOW: 'NDR',
  TURBULENT: 'TRB',
  MULTIPHASE: 'MPH',
  BIT_BOUNCE: 'BBN',
  PACKOFF: 'PKO',
  STICK_SLIP: 'S-S',
  WHIRL: 'WHL',
  FORMATION_CHANGE: 'FMC',
  WASHOUT: 'WSH',
  LOST_CIRCULATION: 'LCR',
  TRANSITION: 'TRN',
  KICK: 'KCK',
  UNKNOWN: 'UNK',
};

/**
 * Polar/radar chart showing transition probability to each regime.
 * Current regime highlighted, trending regime indicated with glow.
 */
export function TransitionRadar({ transition, className }: TransitionRadarProps) {
  const svgW = 300;
  const svgH = 300;
  const cx = svgW / 2;
  const cy = svgH / 2;
  const maxR = 110;

  const { regimes, probPoly, maxProb } = useMemo(() => {
    if (!transition) return { regimes: [] as string[], probPoly: '', maxProb: 0 };

    // Sort by probability descending, take top 8 for readability
    const sorted = Object.entries(transition.probabilities)
      .sort((a, b) => b[1] - a[1]);
    const top = sorted.slice(0, Math.min(12, sorted.length));
    const regimes = top.map(([r]) => r);
    const probs = top.map(([, p]) => p);
    const maxProb = Math.max(...probs, 0.01);

    const n = regimes.length;
    const points = probs.map((p, i) => {
      const angle = (i / n) * Math.PI * 2 - Math.PI / 2;
      const r = (p / maxProb) * maxR;
      const x = cx + Math.cos(angle) * r;
      const y = cy + Math.sin(angle) * r;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');

    return { regimes, probPoly: points, maxProb };
  }, [transition]);

  if (!transition) {
    return (
      <div className={cn('flex items-center justify-center text-muted-foreground text-[10px] font-mono h-full', className)}>
        Awaiting transition probabilities...
      </div>
    );
  }

  const n = regimes.length;
  const riskColor = RISK_COLORS[transition.risk_level] || '#6b7280';

  return (
    <div className={cn('w-full h-full flex flex-col', className)}>
      {/* Header */}
      <div className="text-[9px] font-mono text-muted-foreground uppercase tracking-wider px-2 py-1 shrink-0 flex justify-between">
        <span>Transition Probability</span>
        <span style={{ color: riskColor }}>
          risk: {transition.risk_level}
        </span>
      </div>

      {/* Radar chart */}
      <svg viewBox={`0 0 ${svgW} ${svgH}`} className="w-full flex-1" preserveAspectRatio="xMidYMid meet">
        {/* Grid circles */}
        {[0.25, 0.5, 0.75, 1.0].map(frac => (
          <circle key={frac} cx={cx} cy={cy} r={maxR * frac}
            fill="none" stroke="hsl(var(--border))" strokeWidth={0.5}
            strokeDasharray={frac === 1 ? 'none' : '2,3'} />
        ))}

        {/* Probability labels on grid */}
        {[0.25, 0.5, 0.75, 1.0].map(frac => (
          <text key={`lbl-${frac}`} x={cx + 3} y={cy - maxR * frac - 2}
            fill="hsl(var(--muted-foreground))" fontSize={6} fontFamily="monospace">
            {(maxProb * frac * 100).toFixed(0)}%
          </text>
        ))}

        {/* Axis lines and labels */}
        {regimes.map((regime, i) => {
          const angle = (i / n) * Math.PI * 2 - Math.PI / 2;
          const lx = cx + Math.cos(angle) * maxR;
          const ly = cy + Math.sin(angle) * maxR;
          const tx = cx + Math.cos(angle) * (maxR + 22);
          const ty = cy + Math.sin(angle) * (maxR + 22);
          const isCurrent = regime === transition.current_regime;
          const isTrending = regime === transition.trending_toward;
          const prob = transition.probabilities[regime] ?? 0;

          return (
            <g key={regime}>
              <line x1={cx} y1={cy} x2={lx} y2={ly}
                stroke="hsl(var(--border))" strokeWidth={0.3} />
              {/* Regime label */}
              <text x={tx} y={ty + 3} textAnchor="middle"
                fill={isCurrent ? '#14b8a6' : isTrending ? '#f59e0b' : 'hsl(var(--muted-foreground))'}
                fontSize={isCurrent ? 8 : 7} fontFamily="monospace"
                fontWeight={isCurrent || isTrending ? 'bold' : 'normal'}>
                {REGIME_SHORT[regime] || regime.slice(0, 3)}
              </text>
              {/* Probability value */}
              <text x={tx} y={ty + 12} textAnchor="middle"
                fill="hsl(var(--muted-foreground))" fontSize={6} fontFamily="monospace">
                {(prob * 100).toFixed(1)}%
              </text>
              {/* Trending indicator */}
              {isTrending && !isCurrent && (
                <circle cx={tx} cy={ty - 8} r={3}
                  fill="#f59e0b" fillOpacity={0.6}>
                  <animate attributeName="r" values="2;4;2" dur="2s" repeatCount="indefinite" />
                </circle>
              )}
            </g>
          );
        })}

        {/* Probability polygon */}
        <polygon
          points={probPoly}
          fill="#14b8a6" fillOpacity={0.15}
          stroke="#14b8a6" strokeWidth={1.5}
        />

        {/* Probability dots */}
        {regimes.map((regime, i) => {
          const angle = (i / n) * Math.PI * 2 - Math.PI / 2;
          const prob = transition.probabilities[regime] ?? 0;
          const r = (prob / maxProb) * maxR;
          const x = cx + Math.cos(angle) * r;
          const y = cy + Math.sin(angle) * r;
          const isCurrent = regime === transition.current_regime;
          return (
            <circle key={`dot-${regime}`} cx={x} cy={y}
              r={isCurrent ? 4 : 2.5}
              fill={isCurrent ? '#14b8a6' : '#14b8a6'}
              fillOpacity={isCurrent ? 1 : 0.6}
              stroke={isCurrent ? '#fff' : 'none'} strokeWidth={1} />
          );
        })}

        {/* Center: current regime */}
        <text x={cx} y={cy - 5} textAnchor="middle"
          fill="#14b8a6" fontSize={11} fontFamily="monospace" fontWeight="bold">
          {transition.current_regime}
        </text>
        <text x={cx} y={cy + 8} textAnchor="middle"
          fill="hsl(var(--muted-foreground))" fontSize={7} fontFamily="monospace">
          current regime
        </text>
      </svg>

      {/* Footer stats */}
      <div className="px-2 py-1 text-[8px] font-mono flex gap-3 shrink-0 border-t border-border/30">
        <span className="text-muted-foreground">
          trending: <span className="text-amber-400">{transition.trending_toward}</span>
        </span>
        {transition.estimated_windows_to_transition != null && (
          <span className="text-muted-foreground">
            ETA: <span className="text-foreground/70">{transition.estimated_windows_to_transition} windows</span>
          </span>
        )}
        <span className="text-muted-foreground">
          vel: <span className="text-foreground/70">{transition.velocity_magnitude.toFixed(3)}</span>
        </span>
      </div>
    </div>
  );
}
