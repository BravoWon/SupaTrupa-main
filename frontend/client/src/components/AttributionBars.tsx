import { useMemo } from 'react';
import { cn } from '@/lib/utils';
import type { AttributionResponse } from '@/types';

interface AttributionBarsProps {
  attribution: AttributionResponse | null;
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

const DIM_COLORS: Record<string, string> = {
  betti_0: '#14b8a6',
  betti_1: '#f59e0b',
  entropy_h0: '#8b5cf6',
  entropy_h1: '#ec4899',
  max_lifetime_h0: '#06b6d4',
  max_lifetime_h1: '#f97316',
  mean_lifetime_h0: '#10b981',
  mean_lifetime_h1: '#ef4444',
  n_features_h0: '#6366f1',
  n_features_h1: '#84cc16',
};

/**
 * Horizontal bar chart showing per-TDA-feature contribution to regime classification.
 * Sorted by contribution percentage, showing observed vs signature values.
 */
export function AttributionBars({ attribution, className }: AttributionBarsProps) {
  const sorted = useMemo(() => {
    if (!attribution) return [];
    return [...attribution.attributions].sort((a, b) => b.contribution_pct - a.contribution_pct);
  }, [attribution]);

  if (!attribution) {
    return (
      <div className={cn('flex items-center justify-center text-muted-foreground text-[10px] font-mono h-full', className)}>
        Awaiting attribution data...
      </div>
    );
  }

  return (
    <div className={cn('w-full h-full flex flex-col', className)}>
      {/* Header */}
      <div className="text-[9px] font-mono text-muted-foreground uppercase tracking-wider px-2 py-1 shrink-0 flex justify-between">
        <span>Feature Attribution</span>
        <span className="text-primary">{attribution.regime}</span>
      </div>

      {/* Dominant dimension callout */}
      <div className="px-2 pb-1 text-[9px] font-mono text-foreground/70 shrink-0">
        Dominant: <span className="text-primary">{FEATURE_SHORT[attribution.dominant_dimension] || attribution.dominant_dimension}</span>
        <span className="text-muted-foreground ml-2">
          dist={attribution.total_distance.toFixed(3)}
        </span>
      </div>

      {/* Bar chart */}
      <div className="flex-1 min-h-0 overflow-y-auto px-2 space-y-0.5">
        {sorted.map(attr => {
          const barColor = DIM_COLORS[attr.feature] || '#14b8a6';
          return (
            <div key={attr.feature} className="flex items-center gap-1 text-[9px] font-mono">
              <span className="text-muted-foreground w-14 shrink-0 text-right">
                {FEATURE_SHORT[attr.feature] || attr.feature}
              </span>
              <div className="flex-1 h-2.5 bg-muted/20 rounded-sm overflow-hidden relative">
                <div
                  className="h-full rounded-sm transition-all duration-300"
                  style={{
                    width: `${Math.min(100, attr.contribution_pct)}%`,
                    backgroundColor: barColor,
                    opacity: 0.7,
                  }}
                />
              </div>
              <span className="w-10 text-right text-foreground/70">
                {attr.contribution_pct.toFixed(1)}%
              </span>
            </div>
          );
        })}
      </div>

      {/* Interpretation footer */}
      {attribution.interpretation && (
        <div className="px-2 py-1 text-[8px] font-mono text-muted-foreground leading-tight border-t border-border/30 shrink-0">
          {attribution.interpretation}
        </div>
      )}
    </div>
  );
}
