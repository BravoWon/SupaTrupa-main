import { cn } from '@/lib/utils';

interface InfoTooltipProps {
  text: string;
  className?: string;
}

export function InfoTooltip({ text, className }: InfoTooltipProps) {
  return (
    <span className={cn('relative inline-flex items-center group cursor-help ml-1', className)}>
      <span className="w-3 h-3 rounded-full border border-muted-foreground/50 text-[8px] font-mono flex items-center justify-center text-muted-foreground leading-none">
        ?
      </span>
      <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block z-50 pointer-events-none">
        <span className="bg-card border border-border px-2 py-1 text-[9px] font-mono shadow-lg text-muted-foreground block max-w-[220px] whitespace-normal leading-tight">
          {text}
        </span>
      </span>
    </span>
  );
}
