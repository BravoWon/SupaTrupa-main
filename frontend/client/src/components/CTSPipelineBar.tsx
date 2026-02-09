import { cn } from '@/lib/utils';

type CTSStage = 'environment' | 'sensing' | 'manifold' | 'value' | 'topology' | 'integration' | 'agency';

interface CTSPipelineBarProps {
  activeStage: CTSStage;
  className?: string;
}

const STAGES: { id: CTSStage; symbol: string; label: string }[] = [
  { id: 'environment', symbol: 'E', label: 'Environment' },
  { id: 'sensing', symbol: 's', label: 'Sensing' },
  { id: 'manifold', symbol: 'M', label: 'Manifold' },
  { id: 'value', symbol: 'v', label: 'Value' },
  { id: 'topology', symbol: '\u03A9', label: 'Topology' },
  { id: 'integration', symbol: '\u03A6', label: 'Integration' },
  { id: 'agency', symbol: 'a', label: 'Agency' },
];

export function CTSPipelineBar({ activeStage, className }: CTSPipelineBarProps) {
  return (
    <div
      className={cn(
        'flex items-center justify-center gap-1 border-t border-border bg-card/50 px-4 font-mono text-[10px]',
        className
      )}
    >
      <span className="text-muted-foreground mr-2 hidden sm:inline">CTS PIPELINE</span>
      {STAGES.map((stage, i) => {
        const isActive = stage.id === activeStage;
        return (
          <div key={stage.id} className="flex items-center gap-1">
            <div
              className={cn(
                'w-5 h-5 flex items-center justify-center rounded-sm border transition-colors',
                isActive
                  ? 'border-primary bg-primary/20 text-primary'
                  : 'border-border text-muted-foreground'
              )}
              title={stage.label}
            >
              <span className={cn(isActive && 'animate-pulse')}>{stage.symbol}</span>
            </div>
            {i < STAGES.length - 1 && (
              <span className="text-muted-foreground/40">&rarr;</span>
            )}
          </div>
        );
      })}
    </div>
  );
}

export type { CTSStage };
