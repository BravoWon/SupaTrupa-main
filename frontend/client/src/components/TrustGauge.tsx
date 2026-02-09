import { useMemo } from 'react';

interface TrustGaugeProps {
  confidence: number;
  regime: string;
  className?: string;
}

// Regime severity cost: higher cost = lower autonomy
const REGIME_COST: Record<string, number> = {
  NORMAL: 0.1,
  OPTIMAL: 0.1,
  DARCY_FLOW: 0.15,
  NON_DARCY_FLOW: 0.3,
  FORMATION_CHANGE: 0.35,
  WHIRL: 0.4,
  BIT_BOUNCE: 0.45,
  TRANSITION: 0.35,
  UNKNOWN: 0.5,
  STICK_SLIP: 0.55,
  PACKOFF: 0.6,
  TURBULENT: 0.55,
  MULTIPHASE: 0.6,
  KICK: 0.9,
  WASHOUT: 0.75,
  LOST_CIRCULATION: 0.8,
};

function computeTrust(confidence: number, regime: string): number {
  const V = Math.max(0, Math.min(1, confidence));
  const C = REGIME_COST[regime] ?? 0.5;
  // A proportional V / C, clamped to [0,1]
  const raw = V / (C + 0.1);
  return Math.max(0, Math.min(1, raw / 2)); // normalize so NORMAL+1.0conf ~ 0.9
}

function getTrustLabel(trust: number): { label: string; color: string } {
  if (trust < 0.33) return { label: 'MANUAL', color: '#ef4444' };
  if (trust < 0.66) return { label: 'ASSISTIVE', color: '#eab308' };
  return { label: 'AUTO', color: '#22c55e' };
}

export function TrustGauge({ confidence, regime, className }: TrustGaugeProps) {
  const trust = useMemo(() => computeTrust(confidence, regime), [confidence, regime]);
  const { label, color } = getTrustLabel(trust);

  // SVG semicircular gauge
  const cx = 120;
  const cy = 110;
  const r = 80;
  // Arc from 180deg (left) to 0deg (right) — semicircle
  // Needle angle: trust 0 = left (180deg), trust 1 = right (0deg)
  const needleAngle = Math.PI - trust * Math.PI;
  const needleLen = r - 8;
  const nx = cx + needleLen * Math.cos(needleAngle);
  const ny = cy - needleLen * Math.sin(needleAngle);

  // Arc segments: MANUAL (0-33%), ASSISTIVE (33-66%), AUTO (66-100%)
  const segments = [
    { start: 0, end: 0.33, color: '#ef4444', label: 'MANUAL' },
    { start: 0.33, end: 0.66, color: '#eab308', label: 'ASSISTIVE' },
    { start: 0.66, end: 1.0, color: '#22c55e', label: 'AUTO' },
  ];

  const arcPath = (startPct: number, endPct: number) => {
    const a1 = Math.PI - startPct * Math.PI;
    const a2 = Math.PI - endPct * Math.PI;
    const x1 = cx + r * Math.cos(a1);
    const y1 = cy - r * Math.sin(a1);
    const x2 = cx + r * Math.cos(a2);
    const y2 = cy - r * Math.sin(a2);
    const largeArc = (endPct - startPct) > 0.5 ? 1 : 0;
    return `M ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2}`;
  };

  return (
    <div className={className}>
      <div className="text-[10px] font-mono text-muted-foreground mb-1 uppercase tracking-wider">
        Automation Readiness
      </div>
      <svg viewBox="0 0 240 130" className="w-full max-w-[240px] mx-auto">
        {/* Arc segments */}
        {segments.map(seg => (
          <path
            key={seg.label}
            d={arcPath(seg.start, seg.end)}
            fill="none"
            stroke={seg.color}
            strokeWidth={12}
            opacity={0.3}
            strokeLinecap="round"
          />
        ))}

        {/* Active arc up to current trust */}
        {segments.map(seg => {
          const segStart = seg.start;
          const segEnd = Math.min(seg.end, trust);
          if (segEnd <= segStart) return null;
          return (
            <path
              key={`active-${seg.label}`}
              d={arcPath(segStart, segEnd)}
              fill="none"
              stroke={seg.color}
              strokeWidth={12}
              strokeLinecap="round"
              opacity={0.9}
            />
          );
        })}

        {/* Needle */}
        <line
          x1={cx}
          y1={cy}
          x2={nx}
          y2={ny}
          stroke={color}
          strokeWidth={2.5}
          strokeLinecap="round"
        />
        <circle cx={cx} cy={cy} r={5} fill={color} />
        <circle cx={cx} cy={cy} r={2.5} fill="var(--card)" />

        {/* Labels */}
        <text x={cx - r - 6} y={cy + 16} textAnchor="end" fontSize={8} fill="#ef4444" fontFamily="monospace">
          M
        </text>
        <text x={cx + r + 6} y={cy + 16} textAnchor="start" fontSize={8} fill="#22c55e" fontFamily="monospace">
          A
        </text>

        {/* Center label */}
        <text x={cx} y={cy + 4} textAnchor="middle" fontSize={12} fill={color} fontFamily="monospace" fontWeight="bold">
          {label}
        </text>
        <text x={cx} y={cy + 18} textAnchor="middle" fontSize={9} fill="var(--muted-foreground)" fontFamily="monospace">
          {(trust * 100).toFixed(0)}%
        </text>
      </svg>
    </div>
  );
}
