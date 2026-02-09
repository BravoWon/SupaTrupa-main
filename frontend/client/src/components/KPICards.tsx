import { cn } from '@/lib/utils';
import type { DrillingRecord } from '@/types';

interface KPICardsProps {
  regime: string | null;
  confidence: number | null;
  color: string | null;
  betti: { b0: number; b1: number } | null;
  record: DrillingRecord | null;
  currentDepth: number;
}

function KPICell({ label, value, unit, color }: { label: string; value: string | number; unit?: string; color?: string }) {
  return (
    <div className="bg-card/60 border border-border/50 px-2 py-1.5 min-w-0">
      <div className="text-[9px] font-mono text-muted-foreground uppercase tracking-wider truncate">{label}</div>
      <div className={cn('text-sm font-mono font-medium leading-tight', color)} >
        {value}
        {unit && <span className="text-[9px] text-muted-foreground ml-0.5">{unit}</span>}
      </div>
    </div>
  );
}

function regimeColor(color: string | null): string {
  switch (color) {
    case 'GREEN': return 'text-green-400';
    case 'YELLOW': return 'text-yellow-400';
    case 'ORANGE': return 'text-orange-400';
    case 'RED': return 'text-red-400';
    default: return 'text-foreground';
  }
}

export function KPICards({ regime, confidence, color, betti, record, currentDepth }: KPICardsProps) {
  return (
    <div className="grid grid-cols-2 gap-1">
      <KPICell
        label="Regime"
        value={regime ?? '---'}
        color={regimeColor(color)}
      />
      <KPICell
        label="Confidence"
        value={confidence !== null ? `${(confidence * 100).toFixed(0)}%` : '---'}
        color={confidence !== null && confidence < 0.6 ? 'text-orange-400' : undefined}
      />
      <KPICell label="Drilling Zones" value={betti?.b0 ?? '---'} />
      <KPICell label="Coupling Loops" value={betti?.b1 ?? '---'} />
      <KPICell label="Depth" value={currentDepth.toFixed(1)} unit="ft" />
      <KPICell label="ROP" value={record?.rop?.toFixed(1) ?? '---'} unit="ft/hr" />
      <KPICell label="WOB" value={record?.wob?.toFixed(1) ?? '---'} unit="klb" />
      <KPICell label="RPM" value={record?.rpm?.toFixed(0) ?? '---'} />
      <KPICell label="Torque" value={record?.torque?.toFixed(0) ?? '---'} unit="ft-lb" />
      <KPICell label="SPP" value={record?.spp?.toFixed(0) ?? '---'} unit="psi" />
    </div>
  );
}
