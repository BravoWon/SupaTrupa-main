import { useMemo } from 'react';
import { cn } from '@/lib/utils';
import type { RegimeCompareResponse } from '@/types';

interface RegimeCompareProps {
  comparison: RegimeCompareResponse | null;
  className?: string;
}

const FEATURE_SHORT: Record<string, string> = {
  betti_0: 'Zones',
  betti_1: 'Couplings',
  entropy_h0: 'Zone Stab',
  entropy_h1: 'Coup Stab',
  max_lifetime_h0: 'Dom Zone',
  max_lifetime_h1: 'Str Coup',
  mean_lifetime_h0: 'Avg Zone',
  mean_lifetime_h1: 'Avg Coup',
  n_features_h0: '# Zones',
  n_features_h1: '# Coups',
};

/**
 * Side-by-side radar chart comparing two regime fingerprints.
 * Highlights discriminating features between the two regimes.
 */
export function RegimeCompare({ comparison, className }: RegimeCompareProps) {
  const svgW = 260;
  const svgH = 240;
  const cx = svgW / 2;
  const cy = svgH / 2 - 5;
  const maxR = 85;

  const { polyA, polyB, featureKeys } = useMemo(() => {
    if (!comparison) {
      return { polyA: '', polyB: '', featureKeys: [] as string[], maxVal: 1 };
    }

    const featureKeys = Object.keys(comparison.features_a);
    const n = featureKeys.length;
    if (n === 0) return { polyA: '', polyB: '', featureKeys: [], maxVal: 1 };

    let maxVal = 0;
    for (const key of featureKeys) {
      maxVal = Math.max(maxVal, Math.abs(comparison.features_a[key]));
      maxVal = Math.max(maxVal, Math.abs(comparison.features_b[key]));
    }
    maxVal = maxVal || 1;

    const toPoints = (values: Record<string, number>) =>
      featureKeys
        .map((key, i) => {
          const angle = (i / n) * Math.PI * 2 - Math.PI / 2;
          const r = (Math.abs(values[key]) / maxVal) * maxR;
          const x = cx + Math.cos(angle) * r;
          const y = cy + Math.sin(angle) * r;
          return `${x.toFixed(1)},${y.toFixed(1)}`;
        })
        .join(' ');

    return {
      polyA: toPoints(comparison.features_a),
      polyB: toPoints(comparison.features_b),
      featureKeys,
      maxVal,
    };
  }, [comparison]);

  if (!comparison) {
    return (
      <div className={cn('flex items-center justify-center text-muted-foreground text-[10px] font-mono h-full', className)}>
        Select two regimes to compare...
      </div>
    );
  }

  const n = featureKeys.length;

  return (
    <div className={cn('w-full h-full flex flex-col', className)}>
      {/* Header */}
      <div className="text-[9px] font-mono text-muted-foreground uppercase tracking-wider px-2 py-1 shrink-0 flex justify-between">
        <span>Regime Comparison</span>
        <span className="text-foreground/70">
          d={comparison.topological_distance.toFixed(3)}
        </span>
      </div>

      {/* Radar chart */}
      <svg
        viewBox={`0 0 ${svgW} ${svgH}`}
        className="w-full flex-1"
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Grid circles */}
        {[0.25, 0.5, 0.75, 1.0].map(frac => (
          <circle
            key={frac}
            cx={cx} cy={cy} r={maxR * frac}
            fill="none" stroke="hsl(var(--border))" strokeWidth={0.5}
            strokeDasharray={frac === 1 ? 'none' : '2,3'}
          />
        ))}

        {/* Axis lines and labels */}
        {featureKeys.map((key, i) => {
          const angle = (i / n) * Math.PI * 2 - Math.PI / 2;
          const x = cx + Math.cos(angle) * (maxR + 18);
          const y = cy + Math.sin(angle) * (maxR + 18);
          const lx = cx + Math.cos(angle) * maxR;
          const ly = cy + Math.sin(angle) * maxR;
          const isDiscriminating = comparison.discriminating_features.includes(key);
          return (
            <g key={key}>
              <line x1={cx} y1={cy} x2={lx} y2={ly}
                stroke="hsl(var(--border))" strokeWidth={0.3} />
              <text x={x} y={y + 3} textAnchor="middle"
                className={isDiscriminating ? 'fill-amber-400' : 'fill-muted-foreground'}
                fontSize={6.5} fontFamily="monospace"
                fontWeight={isDiscriminating ? 'bold' : 'normal'}>
                {FEATURE_SHORT[key] || key}
              </text>
            </g>
          );
        })}

        {/* Regime A polygon (teal) */}
        <polygon
          points={polyA}
          fill="#14b8a6" fillOpacity={0.12}
          stroke="#14b8a6" strokeWidth={1.5}
        />

        {/* Regime B polygon (violet) */}
        <polygon
          points={polyB}
          fill="#8b5cf6" fillOpacity={0.12}
          stroke="#8b5cf6" strokeWidth={1.5} strokeDasharray="4,3"
        />

        {/* Legend */}
        <g transform={`translate(4, ${svgH - 20})`}>
          <line x1={0} y1={0} x2={12} y2={0} stroke="#14b8a6" strokeWidth={1.5} />
          <text x={16} y={3} className="fill-foreground" fontSize={7} fontFamily="monospace">
            {comparison.regime_a}
          </text>
          <line x1={100} y1={0} x2={112} y2={0} stroke="#8b5cf6" strokeWidth={1.5} strokeDasharray="3,3" />
          <text x={116} y={3} className="fill-foreground" fontSize={7} fontFamily="monospace">
            {comparison.regime_b}
          </text>
        </g>
      </svg>

      {/* Discriminating features list */}
      {comparison.discriminating_features.length > 0 && (
        <div className="px-2 pb-1 space-y-0.5 shrink-0">
          <div className="text-[8px] font-mono text-muted-foreground uppercase">Discriminating Features</div>
          <div className="flex flex-wrap gap-1">
            {comparison.discriminating_features.map(f => (
              <span key={f} className="text-[8px] font-mono bg-amber-500/15 text-amber-400 px-1 py-0.5 rounded-sm">
                {FEATURE_SHORT[f] || f}
                <span className="text-amber-400/60 ml-0.5">
                  {'\u0394'}{Math.abs(comparison.feature_deltas[f]).toFixed(2)}
                </span>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Interpretation */}
      {comparison.interpretation && (
        <div className="px-2 py-1 text-[8px] font-mono text-muted-foreground leading-tight border-t border-border/30 shrink-0">
          {comparison.interpretation}
        </div>
      )}
    </div>
  );
}
