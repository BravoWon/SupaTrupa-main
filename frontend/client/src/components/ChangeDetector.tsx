import { useEffect, useState } from 'react';
import { cn } from '@/lib/utils';
import type { ChangeDetectResult } from '@/types';

interface ChangeDetectorProps {
  changeResult: ChangeDetectResult | null;
  className?: string;
}

/**
 * Compact bar showing topological change magnitude.
 * Color: green < 0.3, yellow 0.3-0.5, red > 0.5.
 * Flashes when detected_change = true.
 * Sub-bars for landscape_distance and silhouette_distance per dimension.
 */
export function ChangeDetector({ changeResult, className }: ChangeDetectorProps) {
  const [flash, setFlash] = useState(false);

  useEffect(() => {
    if (changeResult?.detected_change) {
      setFlash(true);
      const timer = setTimeout(() => setFlash(false), 1500);
      return () => clearTimeout(timer);
    }
  }, [changeResult?.detected_change, changeResult?.change_magnitude]);

  if (!changeResult) {
    return (
      <div className={cn('flex items-center gap-2 px-3 text-[9px] font-mono text-muted-foreground', className)}>
        <span className="uppercase tracking-wider">Pattern Shift</span>
        <span>--</span>
      </div>
    );
  }

  const mag = changeResult.change_magnitude;
  const barColor = mag < 0.3 ? 'bg-green-500' : mag < 0.5 ? 'bg-yellow-500' : 'bg-red-500';
  const textColor = mag < 0.3 ? 'text-green-400' : mag < 0.5 ? 'text-yellow-400' : 'text-red-400';
  const barWidth = Math.min(100, Math.max(2, mag * 100));

  const landscapeDist = Object.values(changeResult.landscape_distance);
  const silhouetteDist = Object.values(changeResult.silhouette_distance);
  const maxSubBar = Math.max(1, ...landscapeDist, ...silhouetteDist);

  return (
    <div
      className={cn(
        'flex items-center gap-2 px-3 transition-all duration-300',
        flash && 'bg-destructive/20',
        className,
      )}
    >
      {/* Label */}
      <span className="text-[9px] font-mono text-muted-foreground uppercase tracking-wider shrink-0">
        Pattern Shift
      </span>

      {/* Main magnitude bar */}
      <div className="flex items-center gap-1.5 flex-1 min-w-0">
        <div className="w-20 h-2 bg-muted/30 rounded-sm overflow-hidden shrink-0">
          <div
            className={cn('h-full rounded-sm transition-all duration-500', barColor)}
            style={{ width: `${barWidth}%` }}
          />
        </div>
        <span className={cn('text-[10px] font-mono tabular-nums shrink-0', textColor)}>
          {mag.toFixed(3)}
        </span>

        {/* Sub-bars for landscape/silhouette distances */}
        {landscapeDist.length > 0 && (
          <div className="flex items-center gap-1 ml-2">
            <span className="text-[8px] font-mono text-muted-foreground">L:</span>
            {landscapeDist.map((d, i) => (
              <div key={`l-${i}`} className="w-8 h-1.5 bg-muted/30 rounded-sm overflow-hidden">
                <div
                  className="h-full bg-teal-500/70 rounded-sm"
                  style={{ width: `${Math.min(100, (d / maxSubBar) * 100)}%` }}
                />
              </div>
            ))}
          </div>
        )}
        {silhouetteDist.length > 0 && (
          <div className="flex items-center gap-1 ml-1">
            <span className="text-[8px] font-mono text-muted-foreground">S:</span>
            {silhouetteDist.map((d, i) => (
              <div key={`s-${i}`} className="w-8 h-1.5 bg-muted/30 rounded-sm overflow-hidden">
                <div
                  className="h-full bg-amber-500/70 rounded-sm"
                  style={{ width: `${Math.min(100, (d / maxSubBar) * 100)}%` }}
                />
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Change detected indicator */}
      {changeResult.detected_change && (
        <span className={cn(
          'text-[8px] font-mono uppercase px-1 py-0.5 rounded',
          'bg-destructive/20 text-destructive animate-pulse',
        )}>
          CHANGE
        </span>
      )}
    </div>
  );
}
