import { cn } from '@/lib/utils';
import type { ParameterNode, CorrelationEdge } from '@/types';

const CATEGORY_COLORS: Record<string, string> = {
  mechanical: '#f97316',
  hydraulic: '#3b82f6',
  formation: '#a855f7',
  directional: '#06b6d4',
  vibration: '#ec4899',
  performance: '#22c55e',
};

const HEALTH_LABELS: Record<string, { label: string; color: string }> = {
  optimal: { label: 'OPTIMAL', color: 'text-green-400' },
  caution: { label: 'CAUTION', color: 'text-yellow-400' },
  warning: { label: 'WARNING', color: 'text-red-400' },
  critical: { label: 'CRITICAL', color: 'text-fuchsia-400' },
};

interface ParameterDetailCardProps {
  node: ParameterNode;
  edges: CorrelationEdge[];
  allNodes: ParameterNode[];
  recentValues: number[];
  onClose: () => void;
  onChipClick: (nodeId: string) => void;
  className?: string;
}

export function ParameterDetailCard({
  node,
  edges,
  allNodes,
  recentValues,
  onClose,
  onChipClick,
  className,
}: ParameterDetailCardProps) {
  const catColor = CATEGORY_COLORS[node.category] ?? '#888';
  const hl = HEALTH_LABELS[node.health] ?? HEALTH_LABELS.optimal;

  // Connected edges for this node
  const connected = edges.filter((e) => e.source === node.id || e.target === node.id);
  const connectedIds = connected.map((e) => (e.source === node.id ? e.target : e.source));

  // Sparkline
  const sparkW = 180;
  const sparkH = 32;
  const vals = recentValues.length > 0 ? recentValues : [node.current_value];
  const vMin = Math.min(...vals);
  const vMax = Math.max(...vals);
  const range = vMax - vMin || 1;
  const points = vals
    .map((v, i) => {
      const x = (i / Math.max(vals.length - 1, 1)) * sparkW;
      const y = sparkH - ((v - vMin) / range) * sparkH;
      return `${x},${y}`;
    })
    .join(' ');

  return (
    <div
      className={cn(
        'absolute z-50 bg-card border border-border shadow-lg font-mono text-xs w-64',
        className,
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-border/50">
        <div className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: catColor }} />
          <span className="font-bold text-foreground">{node.full_name}</span>
        </div>
        <button
          onClick={onClose}
          className="text-muted-foreground hover:text-foreground text-sm leading-none"
        >
          &times;
        </button>
      </div>

      {/* Body */}
      <div className="px-3 py-2 space-y-2">
        {/* Category + Health */}
        <div className="flex justify-between text-[10px]">
          <span className="text-muted-foreground capitalize">{node.category}</span>
          <span className={hl.color}>{hl.label}</span>
        </div>

        {/* Current / Mean */}
        <div className="grid grid-cols-2 gap-2 text-[10px]">
          <div>
            <span className="text-muted-foreground">Current: </span>
            <span className="text-foreground">
              {node.current_value.toFixed(2)} {node.unit}
            </span>
          </div>
          <div>
            <span className="text-muted-foreground">Mean: </span>
            <span className="text-foreground">
              {node.mean.toFixed(2)} {node.unit}
            </span>
          </div>
        </div>

        {/* Frequency + Correlations count */}
        <div className="grid grid-cols-2 gap-2 text-[10px]">
          <div>
            <span className="text-muted-foreground">Frequency: </span>
            <span className="text-primary">{node.dominant_frequency_hz.toFixed(3)} Hz</span>
          </div>
          <div>
            <span className="text-muted-foreground">Correlations: </span>
            <span className="text-foreground">{connected.length}</span>
          </div>
        </div>

        {/* Z-score */}
        <div className="text-[10px]">
          <span className="text-muted-foreground">Z-score: </span>
          <span
            className={cn(
              Math.abs(node.z_score) > 3 ? 'text-red-400' : Math.abs(node.z_score) > 2 ? 'text-yellow-400' : 'text-foreground',
            )}
          >
            {node.z_score.toFixed(2)}
          </span>
        </div>

        {/* Sparkline */}
        <div className="bg-background/50 border border-border/30 p-1">
          <svg width={sparkW} height={sparkH} className="block">
            <polyline
              points={points}
              fill="none"
              stroke={catColor}
              strokeWidth="1.5"
              strokeLinejoin="round"
            />
          </svg>
        </div>

        {/* Connected params */}
        {connectedIds.length > 0 && (
          <div>
            <div className="text-[9px] text-muted-foreground uppercase tracking-wider mb-1">
              Connected to
            </div>
            <div className="flex flex-wrap gap-1">
              {connectedIds.map((cid) => {
                const cn2 = allNodes.find((n) => n.id === cid);
                const cc = cn2 ? CATEGORY_COLORS[cn2.category] ?? '#888' : '#888';
                const edge = connected.find(
                  (e) => e.source === cid || e.target === cid,
                );
                return (
                  <button
                    key={cid}
                    onClick={() => onChipClick(cid)}
                    className="px-1.5 py-0.5 text-[9px] font-mono border border-border/60 hover:border-primary/60 transition-colors"
                    style={{ borderLeftColor: cc, borderLeftWidth: 2 }}
                  >
                    {cid}
                    {edge && (
                      <span className="text-muted-foreground ml-1">
                        {edge.pearson_r.toFixed(2)}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
