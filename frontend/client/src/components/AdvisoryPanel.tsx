import { useMemo } from 'react';
import { cn } from '@/lib/utils';
import type { AdvisoryResponse, RiskAssessmentResponse } from '@/types';

// Parameter display metadata
const PARAM_META: Record<string, { name: string; unit: string; color: string }> = {
  wob: { name: 'Weight on Bit', unit: 'klbs', color: '#2dd4bf' },
  rpm: { name: 'Rotary Speed', unit: 'rpm', color: '#60a5fa' },
  rop: { name: 'Rate of Penetration', unit: 'ft/hr', color: '#a78bfa' },
  torque: { name: 'Torque', unit: 'kft-lbs', color: '#f59e0b' },
  spp: { name: 'Standpipe Pressure', unit: 'psi', color: '#f472b6' },
};

const RISK_COLORS: Record<string, string> = {
  low: '#22c55e',
  medium: '#eab308',
  high: '#f97316',
  critical: '#ef4444',
};

interface Props {
  advisory: AdvisoryResponse | null;
  risk: RiskAssessmentResponse | null;
}

export function AdvisoryPanel({ advisory, risk }: Props) {
  const riskColor = useMemo(
    () => RISK_COLORS[advisory?.risk_level ?? 'medium'] ?? '#eab308',
    [advisory?.risk_level]
  );

  if (!advisory) {
    return (
      <div className="h-full flex items-center justify-center text-muted-foreground font-mono text-xs">
        <div className="text-center space-y-1">
          <div className="text-sm">ADVISORY ENGINE</div>
          <div className="text-[10px] opacity-60">
            Computing optimal parameter prescription...
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Header: regime transition */}
      <div className="shrink-0 border-b border-border/50 p-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-[9px] font-mono text-muted-foreground uppercase">Advisory</span>
            <span className="text-xs font-mono text-foreground/80">
              {advisory.current_regime}
            </span>
            <span className="text-[10px] text-muted-foreground">&rarr;</span>
            <span className="text-xs font-mono text-primary">
              {advisory.target_regime}
            </span>
          </div>
          {/* Risk badge */}
          <div
            className="px-1.5 py-0.5 text-[9px] font-mono font-bold uppercase tracking-wider"
            style={{
              backgroundColor: `${riskColor}20`,
              color: riskColor,
              border: `1px solid ${riskColor}40`,
            }}
          >
            {advisory.risk_level}
          </div>
        </div>

        {/* Path metrics row */}
        <div className="flex items-center gap-3 mt-1.5">
          <Metric label="Confidence" value={`${(advisory.confidence * 100).toFixed(0)}%`} />
          <Metric label="Geodesic" value={advisory.geodesic_length.toFixed(1)} />
          <Metric label="Efficiency" value={`${advisory.path_efficiency.toFixed(2)}x`} />
          <Metric label="Transitions" value={String(advisory.estimated_transitions)} />
        </div>
      </div>

      {/* Steps: the main prescription */}
      <div className="flex-1 min-h-0 overflow-y-auto p-2 space-y-1.5">
        <div className="text-[9px] font-mono text-muted-foreground uppercase tracking-wider mb-1">
          Parameter Prescription ({advisory.steps.length} steps)
        </div>

        {advisory.steps.map((step) => {
          const meta = PARAM_META[step.parameter] ?? {
            name: step.parameter,
            unit: '',
            color: '#94a3b8',
          };
          const isIncrease = step.change_amount > 0;

          return (
            <div
              key={step.step_index}
              className="bg-card/60 border border-border/40 px-2 py-1.5 space-y-0.5"
            >
              {/* Step header */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5">
                  <span
                    className="text-[9px] font-mono font-bold px-1 py-0.5"
                    style={{
                      backgroundColor: `${meta.color}20`,
                      color: meta.color,
                      border: `1px solid ${meta.color}30`,
                    }}
                  >
                    {step.priority}
                  </span>
                  <span className="text-[11px] font-mono" style={{ color: meta.color }}>
                    {meta.name}
                  </span>
                </div>
                <span
                  className={cn(
                    'text-[10px] font-mono font-bold',
                    isIncrease ? 'text-green-400' : 'text-orange-400'
                  )}
                >
                  {isIncrease ? '+' : ''}
                  {step.change_amount.toFixed(1)} {meta.unit} ({isIncrease ? '+' : ''}
                  {step.change_pct.toFixed(0)}%)
                </span>
              </div>

              {/* Value bar */}
              <div className="flex items-center gap-2">
                <span className="text-[9px] font-mono text-muted-foreground w-14 text-right">
                  {step.current_value.toFixed(1)}
                </span>
                <div className="flex-1 h-1.5 bg-muted/30 relative overflow-hidden">
                  <div
                    className="absolute h-full"
                    style={{
                      backgroundColor: `${meta.color}40`,
                      left: '0',
                      width: `${Math.min(100, Math.abs(step.change_pct))}%`,
                    }}
                  />
                </div>
                <span className="text-[9px] font-mono text-primary w-14">
                  {step.target_value.toFixed(1)}
                </span>
              </div>

              {/* Rationale */}
              <div className="text-[9px] font-mono text-muted-foreground/80 leading-tight">
                {step.rationale}
              </div>
            </div>
          );
        })}

        {/* Reasoning */}
        <div className="mt-2 pt-2 border-t border-border/30">
          <div className="text-[9px] font-mono text-muted-foreground uppercase tracking-wider mb-1">
            Reasoning
          </div>
          {advisory.reasoning.map((r, i) => (
            <div key={i} className="text-[10px] font-mono text-foreground/70 leading-tight mb-0.5">
              {r}
            </div>
          ))}
        </div>
      </div>

      {/* Risk assessment footer */}
      {risk && (
        <div className="shrink-0 border-t border-border/50 p-2 space-y-1">
          <div className="flex items-center justify-between">
            <span className="text-[9px] font-mono text-muted-foreground uppercase tracking-wider">
              Risk Assessment
            </span>
            <div className="flex items-center gap-2">
              <RiskBar label="Regime" value={risk.regime_risk} />
              <RiskBar label="Path" value={risk.path_risk} />
              <RiskBar label="Corr" value={risk.correlation_risk} />
            </div>
          </div>

          {/* Mitigations */}
          {risk.mitigations.length > 0 && (
            <div className="space-y-0.5">
              {risk.mitigations.slice(0, 3).map((m, i) => (
                <div
                  key={i}
                  className="text-[9px] font-mono text-yellow-400/80 leading-tight"
                >
                  &bull; {m}
                </div>
              ))}
            </div>
          )}

          {/* Abort conditions */}
          {risk.abort_conditions.length > 0 && (
            <div className="space-y-0.5 mt-0.5">
              {risk.abort_conditions.slice(0, 2).map((a, i) => (
                <div
                  key={i}
                  className="text-[9px] font-mono text-red-400/70 leading-tight"
                >
                  &times; {a}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/** Compact metric chip */
function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col items-center">
      <span className="text-[8px] font-mono text-muted-foreground uppercase">{label}</span>
      <span className="text-[11px] font-mono text-foreground/90">{value}</span>
    </div>
  );
}

/** Compact risk sub-bar */
function RiskBar({ label, value }: { label: string; value: number }) {
  const pct = Math.min(100, value * 100);
  const color = value < 0.25 ? '#22c55e' : value < 0.5 ? '#eab308' : value < 0.75 ? '#f97316' : '#ef4444';
  return (
    <div className="flex items-center gap-1">
      <span className="text-[8px] font-mono text-muted-foreground">{label}</span>
      <div className="w-8 h-1 bg-muted/30 overflow-hidden">
        <div className="h-full" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}
