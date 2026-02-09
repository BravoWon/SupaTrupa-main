import { cn } from '@/lib/utils';

export type TileStatus = 'green' | 'yellow' | 'orange' | 'red' | 'neutral';
export type TileTrend = 'up' | 'down' | 'stable';

interface DashboardTileProps {
  label: string;
  value: string | number;
  unit?: string;
  status?: TileStatus;
  trend?: TileTrend;
  tooltip?: string;
  onClick?: () => void;
  className?: string;
  large?: boolean;
}

const STATUS_BG: Record<TileStatus, string> = {
  green: 'bg-green-500/10 border-green-500/30',
  yellow: 'bg-yellow-500/10 border-yellow-500/30',
  orange: 'bg-orange-500/10 border-orange-500/30',
  red: 'bg-red-500/10 border-red-500/30',
  neutral: 'bg-card/60 border-border/50',
};

const STATUS_TEXT: Record<TileStatus, string> = {
  green: 'text-green-400',
  yellow: 'text-yellow-400',
  orange: 'text-orange-400',
  red: 'text-red-400',
  neutral: 'text-foreground',
};

const TREND_ICON: Record<TileTrend, string> = {
  up: '\u2191',
  down: '\u2193',
  stable: '\u2192',
};

export function DashboardTile({
  label,
  value,
  unit,
  status = 'neutral',
  trend,
  tooltip,
  onClick,
  className,
  large,
}: DashboardTileProps) {
  return (
    <div
      className={cn(
        'border px-3 py-2 group relative cursor-default transition-colors',
        STATUS_BG[status],
        onClick && 'cursor-pointer hover:brightness-110',
        className
      )}
      onClick={onClick}
      title={tooltip}
    >
      <div className="text-[9px] font-mono text-muted-foreground uppercase tracking-wider truncate">
        {label}
      </div>
      <div className={cn('font-mono font-medium leading-tight flex items-baseline gap-1', large ? 'text-lg' : 'text-sm', STATUS_TEXT[status])}>
        <span>{value}</span>
        {unit && <span className="text-[9px] text-muted-foreground">{unit}</span>}
        {trend && (
          <span
            className={cn(
              'text-[10px]',
              trend === 'up' && 'text-green-400',
              trend === 'down' && 'text-red-400',
              trend === 'stable' && 'text-muted-foreground'
            )}
          >
            {TREND_ICON[trend]}
          </span>
        )}
      </div>
      {tooltip && (
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block z-50 pointer-events-none">
          <div className="bg-card border border-border px-2 py-1 text-[9px] font-mono whitespace-nowrap shadow-lg text-muted-foreground max-w-[200px] text-wrap">
            {tooltip}
          </div>
        </div>
      )}
    </div>
  );
}
