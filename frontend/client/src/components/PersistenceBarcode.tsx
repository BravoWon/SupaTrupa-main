import { useMemo } from 'react';

interface PersistencePair {
  birth: number;
  death: number;
}

interface PersistenceBarcodeProps {
  diagram: {
    h0: PersistencePair[];
    h1: PersistencePair[];
  } | null;
  filtrationRange: [number, number];
  height?: number;
}

const H0_COLOR = 'var(--chart-1)'; // teal
const H1_COLOR = 'var(--chart-4)'; // amber/gold
const BAR_HEIGHT = 6;
const BAR_GAP = 3;
const LABEL_WIDTH = 28;
const RULER_HEIGHT = 20;
const PADDING = 8;

export function PersistenceBarcode({ diagram, filtrationRange, height = 300 }: PersistenceBarcodeProps) {
  // Sort bars by lifetime (longest first), filter infinite (death=-1)
  const { h0Bars, h1Bars } = useMemo(() => {
    if (!diagram) return { h0Bars: [], h1Bars: [] };
    const h0 = diagram.h0
      .filter(p => p.death > 0)
      .map(p => ({ ...p, lifetime: p.death - p.birth }))
      .sort((a, b) => b.lifetime - a.lifetime);
    const h1 = diagram.h1
      .filter(p => p.death > 0)
      .map(p => ({ ...p, lifetime: p.death - p.birth }))
      .sort((a, b) => b.lifetime - a.lifetime);
    return { h0Bars: h0, h1Bars: h1 };
  }, [diagram]);

  const [fMin, fMax] = filtrationRange;
  const fRange = fMax - fMin || 1;

  const chartWidth = 400; // SVG internal width
  const barAreaWidth = chartWidth - LABEL_WIDTH - PADDING * 2;

  const toX = (val: number) => LABEL_WIDTH + ((val - fMin) / fRange) * barAreaWidth;

  const totalH0Height = h0Bars.length * (BAR_HEIGHT + BAR_GAP);
  const totalH1Height = h1Bars.length * (BAR_HEIGHT + BAR_GAP);
  const sectionGap = 12;
  const totalSvgHeight = RULER_HEIGHT + totalH0Height + sectionGap + totalH1Height + PADDING * 2;

  // Filtration ticks
  const ticks = useMemo(() => {
    const count = 6;
    const step = fRange / count;
    return Array.from({ length: count + 1 }, (_, i) => fMin + i * step);
  }, [fMin, fMax, fRange]);

  if (!diagram || (h0Bars.length === 0 && h1Bars.length === 0)) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground text-xs font-mono">
        No persistence data
      </div>
    );
  }

  let y = RULER_HEIGHT + PADDING;

  return (
    <div className="h-full flex flex-col">
      <div className="text-[10px] font-mono text-muted-foreground px-2 pt-1 uppercase tracking-wider shrink-0">
        Feature Lifetime Chart
      </div>
      <div className="flex-1 min-h-0 overflow-y-auto">
        <svg
          viewBox={`0 0 ${chartWidth} ${totalSvgHeight}`}
          className="w-full"
          style={{ minHeight: Math.min(totalSvgHeight, height - 24) }}
          preserveAspectRatio="xMidYMin meet"
        >
          {/* Filtration ruler */}
          <line
            x1={LABEL_WIDTH}
            y1={RULER_HEIGHT - 4}
            x2={LABEL_WIDTH + barAreaWidth}
            y2={RULER_HEIGHT - 4}
            stroke="var(--border)"
            strokeWidth={0.5}
          />
          {ticks.map((t, i) => {
            const x = toX(t);
            return (
              <g key={i}>
                <line x1={x} y1={RULER_HEIGHT - 8} x2={x} y2={RULER_HEIGHT - 2} stroke="var(--muted-foreground)" strokeWidth={0.5} />
                <text x={x} y={RULER_HEIGHT - 10} textAnchor="middle" fontSize={7} fill="var(--muted-foreground)" fontFamily="monospace">
                  {t.toFixed(2)}
                </text>
              </g>
            );
          })}

          {/* H0 section */}
          {h0Bars.length > 0 && (
            <text x={2} y={y + 4} fontSize={8} fill={H0_COLOR} fontFamily="monospace" fontWeight="bold">
              Zones
            </text>
          )}
          {h0Bars.map((bar, i) => {
            const barY = y + i * (BAR_HEIGHT + BAR_GAP);
            const x1 = toX(bar.birth);
            const x2 = toX(bar.death);
            const w = Math.max(1, x2 - x1);
            return (
              <rect
                key={`h0-${i}`}
                x={x1}
                y={barY}
                width={w}
                height={BAR_HEIGHT}
                fill={H0_COLOR}
                opacity={0.8}
                rx={1}
              />
            );
          })}

          {/* H1 section */}
          {(() => {
            const h1Y = y + totalH0Height + sectionGap;
            return (
              <>
                {h1Bars.length > 0 && (
                  <text x={2} y={h1Y + 4} fontSize={8} fill={H1_COLOR} fontFamily="monospace" fontWeight="bold">
                    Couplings
                  </text>
                )}
                {h1Bars.map((bar, i) => {
                  const barY = h1Y + i * (BAR_HEIGHT + BAR_GAP);
                  const x1 = toX(bar.birth);
                  const x2 = toX(bar.death);
                  const w = Math.max(1, x2 - x1);
                  return (
                    <rect
                      key={`h1-${i}`}
                      x={x1}
                      y={barY}
                      width={w}
                      height={BAR_HEIGHT}
                      fill={H1_COLOR}
                      opacity={0.8}
                      rx={1}
                    />
                  );
                })}
              </>
            );
          })()}
        </svg>
      </div>
    </div>
  );
}
