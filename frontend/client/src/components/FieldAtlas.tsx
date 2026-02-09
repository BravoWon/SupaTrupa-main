import { useMemo } from 'react';
import { cn } from '@/lib/utils';
import type { FieldWellEntry, FieldAtlasResponse } from '@/types';

const REGIME_COLORS: Record<string, string> = {
  NORMAL: '#22c55e', OPTIMAL: '#22c55e', DARCY_FLOW: '#22c55e',
  NON_DARCY_FLOW: '#eab308', FORMATION_CHANGE: '#eab308', WHIRL: '#eab308',
  BIT_BOUNCE: '#eab308', TRANSITION: '#eab308', UNKNOWN: '#eab308',
  STICK_SLIP: '#f97316', PACKOFF: '#f97316', TURBULENT: '#f97316', MULTIPHASE: '#f97316',
  KICK: '#ef4444', WASHOUT: '#ef4444', LOST_CIRCULATION: '#ef4444',
};


interface Props {
  atlas: FieldAtlasResponse | null;
  onSelectWell: (wellId: string) => void;
  selectedWellIds: [string | null, string | null];
}

export function FieldAtlas({ atlas, onSelectWell, selectedWellIds }: Props) {
  if (!atlas || atlas.well_count === 0) {
    return (
      <div className="h-full flex items-center justify-center text-muted-foreground font-mono text-xs">
        <div className="text-center space-y-1">
          <div className="text-sm">FIELD ATLAS</div>
          <div className="text-[10px] opacity-60">
            No wells registered. Ingest drilling data to populate.
          </div>
        </div>
      </div>
    );
  }

  const summary = atlas.field_summary;

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Field summary header */}
      <div className="shrink-0 border-b border-border/50 px-2 py-1.5">
        <div className="flex items-center justify-between">
          <span className="text-[9px] font-mono text-muted-foreground uppercase tracking-wider">
            Field Atlas — {atlas.well_count} wells
          </span>
          <div className="flex items-center gap-2">
            <span className="text-[9px] font-mono text-muted-foreground">
              Depth: {summary.depth_range[0].toFixed(0)}–{summary.depth_range[1].toFixed(0)} ft
            </span>
          </div>
        </div>

        {/* Field regime distribution bar */}
        <div className="flex h-1.5 mt-1 overflow-hidden rounded-sm">
          {Object.entries(summary.regime_distribution)
            .sort(([, a], [, b]) => b - a)
            .map(([regime, frac]) => (
              <div
                key={regime}
                className="h-full"
                style={{
                  width: `${frac * 100}%`,
                  backgroundColor: REGIME_COLORS[regime] ?? '#94a3b8',
                  opacity: 0.7,
                }}
                title={`${regime}: ${(frac * 100).toFixed(0)}%`}
              />
            ))}
        </div>
      </div>

      {/* Well cards grid */}
      <div className="flex-1 min-h-0 overflow-y-auto p-2">
        <div className="grid grid-cols-2 gap-1.5">
          {atlas.wells.map((well) => (
            <WellCard
              key={well.well_id}
              well={well}
              meanSig={summary.mean_signature}
              isSelected={selectedWellIds.includes(well.well_id)}
              onClick={() => onSelectWell(well.well_id)}
            />
          ))}
        </div>
      </div>

      {/* Select instructions */}
      <div className="shrink-0 border-t border-border/50 px-2 py-1">
        <span className="text-[8px] font-mono text-muted-foreground">
          Click two wells to compare &bull;{' '}
          {selectedWellIds[0] ? `A: ${selectedWellIds[0].slice(0, 6)}` : 'A: —'}{' '}
          {selectedWellIds[1] ? `B: ${selectedWellIds[1].slice(0, 6)}` : 'B: —'}
        </span>
      </div>
    </div>
  );
}

/** Mini radar chart for a well's TDA fingerprint */
function MiniRadar({ features, meanFeatures }: { features: number[]; meanFeatures: number[] }) {
  const size = 50;
  const cx = size / 2;
  const cy = size / 2;
  const r = size / 2 - 4;
  const n = Math.min(features.length, 10);

  // Normalize features relative to mean (so mean = 0.5 radius)
  const normalized = useMemo(() => {
    return features.slice(0, n).map((v, i) => {
      const mean = meanFeatures[i] ?? 1;
      const scale = Math.max(Math.abs(mean) * 2, 1e-6);
      return Math.min(1, Math.max(0, v / scale));
    });
  }, [features, meanFeatures, n]);

  const points = useMemo(() => {
    return normalized.map((norm, i) => {
      const angle = (i / n) * Math.PI * 2 - Math.PI / 2;
      const dist = norm * r;
      return `${cx + Math.cos(angle) * dist},${cy + Math.sin(angle) * dist}`;
    });
  }, [normalized, n, r, cx, cy]);

  return (
    <svg width={size} height={size} className="shrink-0">
      {/* Grid circles */}
      {[0.33, 0.66, 1].map((f) => (
        <circle
          key={f}
          cx={cx}
          cy={cy}
          r={r * f}
          fill="none"
          stroke="currentColor"
          strokeOpacity={0.08}
          strokeWidth={0.5}
        />
      ))}
      {/* Feature polygon */}
      <polygon
        points={points.join(' ')}
        fill="rgb(45 212 191)"
        fillOpacity={0.2}
        stroke="rgb(45 212 191)"
        strokeWidth={1}
        strokeOpacity={0.6}
      />
    </svg>
  );
}

/** Single well card in the grid */
function WellCard({
  well,
  meanSig,
  isSelected,
  onClick,
}: {
  well: FieldWellEntry;
  meanSig: number[];
  isSelected: boolean;
  onClick: () => void;
}) {
  const regimeColor = REGIME_COLORS[well.regime] ?? '#94a3b8';
  const topRegimes = useMemo(() => {
    return Object.entries(well.regime_distribution)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 3);
  }, [well.regime_distribution]);

  return (
    <button
      onClick={onClick}
      className={cn(
        'bg-card/60 border px-2 py-1.5 text-left transition-colors',
        isSelected
          ? 'border-primary/60 bg-primary/10'
          : 'border-border/40 hover:border-border/80'
      )}
    >
      <div className="flex items-start gap-1.5">
        {/* Mini radar */}
        <MiniRadar features={well.feature_vector} meanFeatures={meanSig} />

        {/* Info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1">
            <div
              className="w-1.5 h-1.5 rounded-full shrink-0"
              style={{ backgroundColor: regimeColor }}
            />
            <span className="text-[10px] font-mono text-foreground truncate">
              {well.name}
            </span>
          </div>

          <div className="text-[8px] font-mono text-muted-foreground mt-0.5">
            {well.depth_min.toFixed(0)}–{well.depth_max.toFixed(0)} ft &bull; {well.num_records} pts
          </div>

          <div className="flex items-center gap-1 mt-0.5">
            <span
              className="text-[8px] font-mono font-bold"
              style={{ color: regimeColor }}
            >
              {well.regime}
            </span>
            <span className="text-[8px] font-mono text-muted-foreground">
              {(well.confidence * 100).toFixed(0)}%
            </span>
          </div>

          {/* Regime distribution mini-bar */}
          <div className="flex h-1 mt-0.5 overflow-hidden rounded-sm">
            {topRegimes.map(([regime, frac]) => (
              <div
                key={regime}
                className="h-full"
                style={{
                  width: `${frac * 100}%`,
                  backgroundColor: REGIME_COLORS[regime] ?? '#94a3b8',
                  opacity: 0.7,
                }}
              />
            ))}
          </div>
        </div>
      </div>
    </button>
  );
}
