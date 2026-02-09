import { useMemo } from 'react';
import { cn } from '@/lib/utils';
import type { AnalyzerHistoryEntry } from '@/types';

interface RegimeStripProps {
  history: AnalyzerHistoryEntry[];
  currentIndex?: number;
  className?: string;
}

const COLOR_MAP: Record<string, string> = {
  GREEN: 'bg-green-500',
  YELLOW: 'bg-yellow-500',
  ORANGE: 'bg-orange-500',
  RED: 'bg-red-500',
};

export function RegimeStrip({ history, currentIndex, className }: RegimeStripProps) {
  const cells = useMemo(() => {
    if (history.length === 0) return [];
    return history.map((entry, i) => ({
      ...entry,
      index: i,
      colorClass: COLOR_MAP[entry.color] || 'bg-muted',
    }));
  }, [history]);

  if (cells.length === 0) {
    return (
      <div className={cn('h-full flex items-center justify-center', className)}>
        <span className="text-[9px] font-mono text-muted-foreground">
          Step through depth to build regime strip
        </span>
      </div>
    );
  }

  return (
    <div className={cn('h-full flex items-center px-2 gap-px', className)}>
      <span className="text-[9px] font-mono text-muted-foreground shrink-0 mr-1">
        REGIME STRIP
      </span>
      <div className="flex-1 flex h-5 gap-px overflow-hidden">
        {cells.map((cell) => (
          <div
            key={cell.index}
            className={cn(
              'flex-1 min-w-[3px] transition-opacity cursor-default group relative',
              cell.colorClass,
              currentIndex === cell.index ? 'ring-1 ring-white' : 'opacity-80 hover:opacity-100'
            )}
            title={`${cell.start.toFixed(0)}–${cell.end.toFixed(0)} ft | ${cell.regime} (${(cell.confidence * 100).toFixed(0)}%)`}
          >
            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block z-50 pointer-events-none">
              <div className="bg-card border border-border px-2 py-1 text-[9px] font-mono whitespace-nowrap shadow-lg">
                <div>{cell.start.toFixed(0)}–{cell.end.toFixed(0)} ft</div>
                <div className="text-foreground">{cell.regime}</div>
                <div className="text-muted-foreground">{(cell.confidence * 100).toFixed(0)}%</div>
              </div>
            </div>
          </div>
        ))}
      </div>
      <span className="text-[9px] font-mono text-muted-foreground shrink-0 ml-1">
        {cells.length} windows
      </span>
    </div>
  );
}
