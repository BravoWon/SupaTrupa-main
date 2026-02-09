import { cn } from '@/lib/utils';
import type { NetworkStats } from '@/types';

const HEALTH_DOT: Record<string, string> = {
  optimal: 'bg-green-500',
  caution: 'bg-yellow-500',
  warning: 'bg-red-500',
  critical: 'bg-fuchsia-500',
};

interface NetworkStatsBarProps {
  stats: NetworkStats | null;
  className?: string;
}

export function NetworkStatsBar({ stats, className }: NetworkStatsBarProps) {
  if (!stats) {
    return (
      <div
        className={cn(
          'flex items-center justify-center gap-3 border-t border-border bg-card/50 px-4 font-mono text-[10px] text-muted-foreground',
          className,
        )}
      >
        No network data
      </div>
    );
  }

  const items: { label: string; value: string | number; warn?: boolean }[] = [
    { label: 'Nodes', value: stats.nodeCount },
    { label: 'Correlations', value: stats.edgeCount },
    { label: 'Strong (r>0.6)', value: stats.strongCount },
    {
      label: 'Anomalies',
      value: stats.anomalyCount,
      warn: stats.anomalyCount > 0,
    },
    { label: 'Update', value: `${stats.computationTimeMs.toFixed(1)}ms` },
  ];

  const dotClass = HEALTH_DOT[stats.systemHealth] ?? HEALTH_DOT.optimal;

  return (
    <div
      className={cn(
        'flex items-center justify-center gap-4 border-t border-border bg-card/50 px-4 font-mono text-[10px]',
        className,
      )}
    >
      {items.map((item, i) => (
        <span key={i} className={item.warn ? 'text-orange-400' : 'text-muted-foreground'}>
          <span className="text-foreground/70">{item.value}</span>{' '}
          {item.label}
          {i < items.length - 1 && (
            <span className="ml-4 text-border">|</span>
          )}
        </span>
      ))}
      <span className="flex items-center gap-1.5 ml-2">
        <span className={cn('w-2 h-2 rounded-full', dotClass)} />
        <span className="text-foreground/80 uppercase">{stats.systemHealth}</span>
      </span>
    </div>
  );
}
