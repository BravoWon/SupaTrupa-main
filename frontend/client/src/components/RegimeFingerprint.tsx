import { useMemo } from 'react';
import { cn } from '@/lib/utils';
import type { FingerprintResponse } from '@/types';

interface RegimeFingerprintProps {
  fingerprint: FingerprintResponse | null;
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
 * Radar-style chart showing observed TDA features vs matched regime signature.
 * Highlights which features diverge most from the regime fingerprint.
 */
export function RegimeFingerprint({ fingerprint, className }: RegimeFingerprintProps) {
  const svgW = 260;
  const svgH = 260;
  const cx = svgW / 2;
  const cy = svgH / 2;
  const maxR = 100;

  const { obsPoints, sigPoints, featureKeys } = useMemo(() => {
    if (!fingerprint) {
      return { obsPoints: '', sigPoints: '', featureKeys: [] as string[], maxVal: 1 };
    }

    const featureKeys = Object.keys(fingerprint.observed_features);
    const n = featureKeys.length;
    if (n === 0) return { obsPoints: '', sigPoints: '', featureKeys: [], maxVal: 1 };

    // Find max across both observed and signature for normalization
    let maxVal = 0;
    for (const key of featureKeys) {
      maxVal = Math.max(maxVal, Math.abs(fingerprint.observed_features[key]));
      maxVal = Math.max(maxVal, Math.abs(fingerprint.matched_signature[key]));
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
      obsPoints: toPoints(fingerprint.observed_features),
      sigPoints: toPoints(fingerprint.matched_signature),
      featureKeys,
      maxVal,
    };
  }, [fingerprint]);

  if (!fingerprint) {
    return (
      <div className={cn('flex items-center justify-center text-muted-foreground text-[10px] font-mono h-full', className)}>
        Awaiting fingerprint data...
      </div>
    );
  }

  const n = featureKeys.length;

  return (
    <div className={cn('w-full h-full flex flex-col', className)}>
      <div className="text-[9px] font-mono text-muted-foreground uppercase tracking-wider px-2 py-1 shrink-0 flex justify-between">
        <span>Regime Fingerprint</span>
        <span className="text-primary">{fingerprint.matched_regime}</span>
      </div>
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
          return (
            <g key={key}>
              <line x1={cx} y1={cy} x2={lx} y2={ly}
                stroke="hsl(var(--border))" strokeWidth={0.3} />
              <text x={x} y={y + 3} textAnchor="middle"
                className="fill-muted-foreground" fontSize={6.5} fontFamily="monospace">
                {FEATURE_SHORT[key] || key}
              </text>
            </g>
          );
        })}

        {/* Signature polygon (filled) */}
        <polygon
          points={sigPoints}
          fill="#14b8a6" fillOpacity={0.12}
          stroke="#14b8a6" strokeWidth={1} strokeDasharray="4,3"
        />

        {/* Observed polygon (filled) */}
        <polygon
          points={obsPoints}
          fill="#f59e0b" fillOpacity={0.15}
          stroke="#f59e0b" strokeWidth={1.5}
        />

        {/* Legend */}
        <g transform={`translate(4, ${svgH - 20})`}>
          <line x1={0} y1={0} x2={12} y2={0} stroke="#f59e0b" strokeWidth={1.5} />
          <text x={16} y={3} className="fill-foreground" fontSize={7} fontFamily="monospace">
            Observed
          </text>
          <line x1={70} y1={0} x2={82} y2={0} stroke="#14b8a6" strokeWidth={1} strokeDasharray="3,3" />
          <text x={86} y={3} className="fill-foreground" fontSize={7} fontFamily="monospace">
            Signature
          </text>
        </g>
      </svg>

      {/* Top drivers */}
      {fingerprint.top_drivers.length > 0 && (
        <div className="px-2 pb-1 space-y-0.5">
          <div className="text-[8px] font-mono text-muted-foreground uppercase">Top Drivers</div>
          {fingerprint.top_drivers.slice(0, 3).map(d => (
            <div key={d.feature} className="flex items-center gap-1 text-[9px] font-mono">
              <span className="text-muted-foreground w-16 shrink-0">
                {FEATURE_SHORT[d.feature] || d.feature}
              </span>
              <div className="flex-1 h-1.5 bg-muted/30 rounded-sm overflow-hidden">
                <div
                  className="h-full bg-amber-500/70 rounded-sm"
                  style={{ width: `${Math.min(100, d.contribution_pct)}%` }}
                />
              </div>
              <span className="text-foreground/70 w-10 text-right">{d.contribution_pct}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
