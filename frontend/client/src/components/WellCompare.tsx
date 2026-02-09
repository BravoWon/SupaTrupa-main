import { useMemo } from 'react';
import type { FieldCompareResponse, FieldWellEntry } from '@/types';

const FEATURE_NAMES = [
  'betti_0', 'betti_1', 'entropy_h0', 'entropy_h1',
  'max_lifetime_h0', 'max_lifetime_h1', 'mean_lifetime_h0', 'mean_lifetime_h1',
  'n_features_h0', 'n_features_h1',
];

const FEATURE_SHORT: Record<string, string> = {
  betti_0: 'Drilling Zones', betti_1: 'Coupling Loops',
  entropy_h0: 'Zone Stability', entropy_h1: 'Coupling Stability',
  max_lifetime_h0: 'Dominant Zone', max_lifetime_h1: 'Strongest Coupling',
  mean_lifetime_h0: 'Avg Zone', mean_lifetime_h1: 'Avg Coupling',
  n_features_h0: 'Total Zones', n_features_h1: 'Total Couplings',
};

interface Props {
  comparison: FieldCompareResponse | null;
  wellA: FieldWellEntry | null;
  wellB: FieldWellEntry | null;
}

const W = 500;
const H = 350;
const PAD = { top: 25, right: 20, bottom: 20, left: 100 };

export function WellCompare({ comparison, wellA, wellB }: Props) {
  if (!comparison || !wellA || !wellB) {
    return (
      <div className="h-full flex items-center justify-center text-muted-foreground font-mono text-xs">
        <div className="text-center space-y-1">
          <div className="text-sm">WELL COMPARE</div>
          <div className="text-[10px] opacity-60">
            Select two wells in the atlas to compare
          </div>
        </div>
      </div>
    );
  }

  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const barH = innerH / FEATURE_NAMES.length;

  // Compute max magnitude across both wells for scaling
  const maxVal = useMemo(() => {
    let max = 1;
    for (const f of FEATURE_NAMES) {
      const i = FEATURE_NAMES.indexOf(f);
      max = Math.max(max, Math.abs(wellA.feature_vector[i]), Math.abs(wellB.feature_vector[i]));
    }
    return max;
  }, [wellA, wellB]);

  const colorA = '#2dd4bf'; // teal
  const colorB = '#a78bfa'; // violet

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Header */}
      <div className="shrink-0 border-b border-border/50 px-2 py-1.5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-sm" style={{ backgroundColor: colorA }} />
              <span className="text-[10px] font-mono" style={{ color: colorA }}>
                {wellA.name}
              </span>
            </div>
            <span className="text-[9px] text-muted-foreground">vs</span>
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-sm" style={{ backgroundColor: colorB }} />
              <span className="text-[10px] font-mono" style={{ color: colorB }}>
                {wellB.name}
              </span>
            </div>
          </div>
          <span className="text-[9px] font-mono text-muted-foreground">
            dist: {comparison.topological_distance.toFixed(2)}
          </span>
        </div>
      </div>

      {/* Butterfly bar chart */}
      <div className="flex-1 min-h-0 relative">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="w-full h-full"
          preserveAspectRatio="xMidYMid meet"
        >
          {/* Center axis */}
          <line
            x1={PAD.left + innerW / 2}
            y1={PAD.top}
            x2={PAD.left + innerW / 2}
            y2={PAD.top + innerH}
            stroke="currentColor"
            strokeOpacity={0.15}
          />

          {FEATURE_NAMES.map((fname, i) => {
            const y = PAD.top + i * barH;
            const valA = wellA.feature_vector[i];
            const valB = wellB.feature_vector[i];
            const normA = valA / maxVal;
            const normB = valB / maxVal;
            const halfW = innerW / 2;
            const centerX = PAD.left + halfW;
            const isDiscriminating = comparison.discriminating_features.includes(fname);

            // Bar A goes left from center
            const barWidthA = Math.abs(normA) * halfW * 0.85;
            // Bar B goes right from center
            const barWidthB = Math.abs(normB) * halfW * 0.85;

            return (
              <g key={fname}>
                {/* Row background for discriminating features */}
                {isDiscriminating && (
                  <rect
                    x={PAD.left}
                    y={y}
                    width={innerW}
                    height={barH}
                    fill="#f59e0b"
                    fillOpacity={0.06}
                  />
                )}

                {/* Feature label */}
                <text
                  x={PAD.left - 4}
                  y={y + barH / 2}
                  textAnchor="end"
                  dominantBaseline="middle"
                  className="text-[8px] font-mono"
                  fill={isDiscriminating ? '#f59e0b' : 'currentColor'}
                  fillOpacity={isDiscriminating ? 0.9 : 0.5}
                >
                  {FEATURE_SHORT[fname] ?? fname}
                </text>

                {/* Bar A (left, teal) */}
                <rect
                  x={centerX - barWidthA}
                  y={y + 2}
                  width={barWidthA}
                  height={barH - 4}
                  fill={colorA}
                  fillOpacity={0.5}
                  rx={1}
                />
                {/* Value label A */}
                <text
                  x={centerX - barWidthA - 2}
                  y={y + barH / 2}
                  textAnchor="end"
                  dominantBaseline="middle"
                  className="text-[7px] font-mono"
                  fill={colorA}
                  fillOpacity={0.8}
                >
                  {valA.toFixed(1)}
                </text>

                {/* Bar B (right, violet) */}
                <rect
                  x={centerX}
                  y={y + 2}
                  width={barWidthB}
                  height={barH - 4}
                  fill={colorB}
                  fillOpacity={0.5}
                  rx={1}
                />
                {/* Value label B */}
                <text
                  x={centerX + barWidthB + 2}
                  y={y + barH / 2}
                  textAnchor="start"
                  dominantBaseline="middle"
                  className="text-[7px] font-mono"
                  fill={colorB}
                  fillOpacity={0.8}
                >
                  {valB.toFixed(1)}
                </text>

                {/* Horizontal grid */}
                <line
                  x1={PAD.left}
                  y1={y + barH}
                  x2={PAD.left + innerW}
                  y2={y + barH}
                  stroke="currentColor"
                  strokeOpacity={0.04}
                />
              </g>
            );
          })}
        </svg>
      </div>

      {/* Metrics footer */}
      <div className="shrink-0 border-t border-border/50 px-2 py-1.5 space-y-1">
        <div className="flex items-center gap-3">
          <MetricChip
            label="Regime Sim"
            value={`${(comparison.regime_similarity * 100).toFixed(0)}%`}
            color={comparison.regime_similarity > 0.5 ? '#22c55e' : '#f97316'}
          />
          <MetricChip
            label="Depth Overlap"
            value={`${(comparison.depth_overlap * 100).toFixed(0)}%`}
            color={comparison.depth_overlap > 0.5 ? '#22c55e' : '#eab308'}
          />
          <MetricChip
            label="Topo Dist"
            value={comparison.topological_distance.toFixed(2)}
            color={comparison.topological_distance < 1 ? '#22c55e' : comparison.topological_distance < 3 ? '#eab308' : '#ef4444'}
          />
        </div>

        <div className="text-[9px] font-mono text-foreground/60 leading-tight">
          {comparison.interpretation}
        </div>
      </div>
    </div>
  );
}

function MetricChip({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="flex flex-col items-center">
      <span className="text-[7px] font-mono text-muted-foreground uppercase">{label}</span>
      <span className="text-[10px] font-mono font-bold" style={{ color }}>
        {value}
      </span>
    </div>
  );
}
