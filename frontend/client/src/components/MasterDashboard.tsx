import { cn } from '@/lib/utils';
import { DashboardTile, type TileStatus } from '@/components/DashboardTile';
import type { DashboardSummary } from '@/types';

interface MasterDashboardProps {
  data: DashboardSummary | null;
  error?: string | null;
  onNavigate?: (tab: string) => void;
  className?: string;
}

function colorToStatus(color: string): TileStatus {
  switch (color) {
    case 'GREEN':
      return 'green';
    case 'YELLOW':
      return 'yellow';
    case 'ORANGE':
      return 'orange';
    case 'RED':
      return 'red';
    default:
      return 'neutral';
  }
}

function riskToStatus(risk: string): TileStatus {
  switch (risk) {
    case 'low':
      return 'green';
    case 'medium':
      return 'yellow';
    case 'high':
      return 'orange';
    case 'critical':
      return 'red';
    default:
      return 'neutral';
  }
}

function confidenceToStatus(c: number): TileStatus {
  if (c >= 0.8) return 'green';
  if (c >= 0.6) return 'yellow';
  if (c >= 0.4) return 'orange';
  return 'red';
}

function predictabilityToStatus(p: number): TileStatus {
  if (p >= 0.7) return 'green';
  if (p >= 0.4) return 'yellow';
  if (p >= 0.2) return 'orange';
  return 'red';
}

function stabilityToStatus(s: number): TileStatus {
  if (s < 0.5) return 'green';
  if (s < 1.0) return 'yellow';
  if (s < 2.0) return 'orange';
  return 'red';
}

export function MasterDashboard({ data, error, onNavigate, className }: MasterDashboardProps) {
  if (!data) {
    if (error) {
      return (
        <div className={cn('h-full flex items-center justify-center', className)}>
          <div className="flex flex-col items-center gap-3 max-w-md text-center">
            <div className="w-10 h-10 rounded-full border-2 border-red-500/40 flex items-center justify-center">
              <span className="text-red-400 text-lg">!</span>
            </div>
            <span className="text-xs font-mono text-red-400">{error}</span>
            <span className="text-[10px] font-mono text-muted-foreground">
              Switch to another tab and back to retry
            </span>
          </div>
        </div>
      );
    }
    return (
      <div className={cn('h-full flex items-center justify-center', className)}>
        <div className="flex flex-col items-center gap-3">
          <div className="w-10 h-10 border-2 border-primary border-t-transparent rounded-full animate-spin" />
          <span className="text-xs font-mono text-muted-foreground">Loading dashboard...</span>
        </div>
      </div>
    );
  }

  const regimeStatus = colorToStatus(data.color);
  const confStatus = confidenceToStatus(data.confidence);
  const riskStatus = riskToStatus(data.transition_risk);
  const predStatus = predictabilityToStatus(data.predictability_index);
  const stabStatus = stabilityToStatus(data.signature_stability);
  const consistStatus = predictabilityToStatus(data.behavioral_consistency);

  return (
    <div className={cn('h-full flex flex-col p-4 gap-4 overflow-y-auto', className)}>
      {/* Section 1: Drilling Status */}
      <div>
        <div className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest mb-2">
          Drilling Status
        </div>
        <div className="grid grid-cols-4 gap-2">
          <DashboardTile
            label="Current Regime"
            value={data.regime_display}
            status={regimeStatus}
            large
            tooltip="Active drilling regime classified from parameter topology"
          />
          <DashboardTile
            label="Confidence"
            value={`${(data.confidence * 100).toFixed(0)}%`}
            status={confStatus}
            large
            tooltip="Classification confidence — below 60% indicates ambiguous regime"
          />
          <DashboardTile
            label="Risk Level"
            value={data.transition_risk.toUpperCase()}
            status={riskStatus}
            large
            tooltip="Risk of imminent regime transition based on topological trajectory"
            onClick={() => onNavigate?.('forecast')}
          />
          <DashboardTile
            label="Trending Toward"
            value={data.trending_toward}
            status={data.trending_toward === data.regime_display ? 'green' : 'yellow'}
            large
            tooltip="Predicted next regime based on pattern evolution"
            onClick={() => onNavigate?.('forecast')}
          />
        </div>
      </div>

      {/* Section 2: Parameters + Pattern Analysis side by side */}
      <div className="grid grid-cols-2 gap-4">
        {/* Left: Drilling Parameters */}
        <div>
          <div className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest mb-2">
            Drilling Parameters
          </div>
          <div className="grid grid-cols-3 gap-2">
            <DashboardTile
              label="Rate of Penetration"
              value={data.rop.toFixed(1)}
              unit="ft/hr"
              status="neutral"
              tooltip="Current rate of penetration"
            />
            <DashboardTile
              label="Weight on Bit"
              value={data.wob.toFixed(1)}
              unit="klb"
              status="neutral"
              tooltip="Current weight on bit"
            />
            <DashboardTile
              label="Rotary Speed"
              value={data.rpm.toFixed(0)}
              unit="RPM"
              status="neutral"
              tooltip="Current rotary speed (RPM)"
            />
          </div>
          <div className="grid grid-cols-2 gap-2 mt-2">
            <DashboardTile
              label="Torque"
              value={data.torque.toFixed(0)}
              unit="ft-lb"
              status="neutral"
              tooltip="Current surface torque"
            />
            <DashboardTile
              label="Standpipe Pressure"
              value={data.spp.toFixed(0)}
              unit="psi"
              status="neutral"
              tooltip="Current standpipe pressure"
            />
          </div>
        </div>

        {/* Right: Pattern Analysis */}
        <div>
          <div className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest mb-2">
            Pattern Analysis
          </div>
          <div className="grid grid-cols-3 gap-2">
            <DashboardTile
              label="Drilling Zones"
              value={data.drilling_zones}
              status="neutral"
              tooltip="Number of distinct operational zones detected in parameter space (connected components)"
              onClick={() => onNavigate?.('fingerprint')}
            />
            <DashboardTile
              label="Coupling Loops"
              value={data.coupling_loops}
              status="neutral"
              tooltip="Number of cyclic parameter couplings detected (feedback loops between drilling parameters)"
              onClick={() => onNavigate?.('fingerprint')}
            />
            <DashboardTile
              label="Signature Stability"
              value={data.signature_stability.toFixed(2)}
              status={stabStatus}
              tooltip="Lower values indicate more stable, predictable drilling patterns"
              onClick={() => onNavigate?.('shadow')}
            />
          </div>
          <div className="grid grid-cols-3 gap-2 mt-2">
            <DashboardTile
              label="Predictability"
              value={`${(data.predictability_index * 100).toFixed(0)}%`}
              status={predStatus}
              tooltip="How predictable is current drilling behavior — high = steady state, low = chaotic"
              onClick={() => onNavigate?.('shadow')}
            />
            <DashboardTile
              label="Consistency"
              value={`${(data.behavioral_consistency * 100).toFixed(0)}%`}
              status={consistStatus}
              tooltip="How repeatable are current drilling dynamics — high = deterministic patterns"
              onClick={() => onNavigate?.('shadow')}
            />
            <DashboardTile
              label="Est. Windows"
              value={data.estimated_windows_to_transition !== null ? data.estimated_windows_to_transition : 'N/A'}
              status={data.estimated_windows_to_transition !== null && data.estimated_windows_to_transition < 5 ? 'orange' : 'neutral'}
              tooltip="Estimated number of analysis windows before regime transition"
              onClick={() => onNavigate?.('forecast')}
            />
          </div>
        </div>
      </div>

      {/* Section 3: Recommended Action */}
      <div>
        <div className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest mb-2">
          Recommended Action
        </div>
        <div className={cn('border px-4 py-3', regimeStatus === 'green' ? 'bg-green-500/5 border-green-500/20' : regimeStatus === 'red' ? 'bg-red-500/5 border-red-500/20' : 'bg-card/60 border-border/50')}>
          <div className="text-sm font-mono leading-relaxed text-foreground/90 mb-2">
            {data.recommendation}
          </div>
          {data.top_advisory && (
            <div className="flex items-center gap-3 mt-2">
              <span className="text-[10px] font-mono text-primary px-2 py-0.5 bg-primary/10 border border-primary/30">
                TOP ADVISORY: {data.top_advisory}
              </span>
              {data.advisory_risk && (
                <span className={cn(
                  'text-[10px] font-mono px-2 py-0.5 border',
                  data.advisory_risk === 'low' && 'text-green-400 border-green-500/30',
                  data.advisory_risk === 'medium' && 'text-yellow-400 border-yellow-500/30',
                  data.advisory_risk === 'high' && 'text-orange-400 border-orange-500/30',
                  data.advisory_risk === 'critical' && 'text-red-400 border-red-500/30',
                )}>
                  RISK: {data.advisory_risk.toUpperCase()}
                </span>
              )}
              <button
                onClick={() => onNavigate?.('advisory')}
                className="text-[10px] font-mono text-primary hover:underline ml-auto"
              >
                VIEW FULL ADVISORY &rarr;
              </button>
            </div>
          )}
          {!data.top_advisory && (
            <button
              onClick={() => onNavigate?.('advisory')}
              className="text-[10px] font-mono text-primary hover:underline"
            >
              VIEW ADVISORY PANEL &rarr;
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
