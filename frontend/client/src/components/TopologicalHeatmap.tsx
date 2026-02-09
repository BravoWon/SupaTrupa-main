import { useRef, useEffect, useMemo } from 'react';
import { cn } from '@/lib/utils';
import type { WindowedSignature } from '@/types';

interface TopologicalHeatmapProps {
  windows: WindowedSignature[];
  className?: string;
}

/**
 * Canvas heatmap: X = window index, Y = feature dimension, color = intensity.
 * Shows how topological features evolve across sliding windows.
 * Regime transitions appear as color boundary shifts.
 */
export function TopologicalHeatmap({ windows, className }: TopologicalHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Build the heatmap matrix: rows = features, cols = windows
  const { matrix, featureLabels, maxVal } = useMemo(() => {
    if (windows.length < 2) {
      return { matrix: [] as number[][], featureLabels: [] as string[], maxVal: 1 };
    }

    const featureLabels = ['Zones', 'Couplings', 'Zone Stab', 'Coup Stab', 'Zone Pers', 'Coup Pers'];
    const numFeatures = featureLabels.length;
    const numWindows = windows.length;

    const matrix: number[][] = Array.from({ length: numFeatures }, () =>
      new Array(numWindows).fill(0)
    );

    let maxVal = 0;
    for (let j = 0; j < numWindows; j++) {
      const w = windows[j];
      const vals = [
        w.betti_0,
        w.betti_1,
        w.entropy_h0,
        w.entropy_h1,
        w.total_persistence_h0,
        w.total_persistence_h1,
      ];
      for (let i = 0; i < numFeatures; i++) {
        matrix[i][j] = vals[i];
        if (vals[i] > maxVal) maxVal = vals[i];
      }
    }

    return { matrix, featureLabels, maxVal: maxVal || 1 };
  }, [windows]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || matrix.length === 0) return;

    const rect = container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = rect.width;
    const h = rect.height;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    const labelW = 48;
    const padTop = 4;
    const padBottom = 16;
    const plotW = w - labelW - 8;
    const plotH = h - padTop - padBottom;

    const numFeatures = matrix.length;
    const numWindows = matrix[0].length;
    const cellW = plotW / numWindows;
    const cellH = plotH / numFeatures;

    // Clear
    ctx.fillStyle = 'hsl(222, 30%, 6%)';
    ctx.fillRect(0, 0, w, h);

    // Normalize per-row for better contrast
    for (let row = 0; row < numFeatures; row++) {
      const rowMax = Math.max(1e-10, ...matrix[row]);
      for (let col = 0; col < numWindows; col++) {
        const intensity = matrix[row][col] / rowMax;
        const x = labelW + col * cellW;
        const y = padTop + row * cellH;

        // Color ramp: dark teal -> bright teal -> white
        const r = Math.round(20 + intensity * 200);
        const g = Math.round(40 + intensity * 215);
        const b = Math.round(50 + intensity * 180);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(x, y, cellW + 0.5, cellH + 0.5);
      }
    }

    // Row labels
    ctx.fillStyle = 'hsl(210, 15%, 55%)';
    ctx.font = '8px monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < numFeatures; i++) {
      const y = padTop + i * cellH + cellH / 2;
      ctx.fillText(featureLabels[i], labelW - 4, y);
    }

    // X-axis markers
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const step = Math.max(1, Math.floor(numWindows / 6));
    for (let j = 0; j < numWindows; j += step) {
      const x = labelW + j * cellW + cellW / 2;
      ctx.fillText(`${j}`, x, padTop + plotH + 2);
    }
  }, [matrix, featureLabels, maxVal]);

  if (windows.length < 2) {
    return (
      <div className={cn('flex items-center justify-center text-muted-foreground text-[10px] font-mono h-full', className)}>
        Awaiting windowed data...
      </div>
    );
  }

  return (
    <div className={cn('w-full h-full flex flex-col', className)}>
      <div className="text-[9px] font-mono text-muted-foreground uppercase tracking-wider px-2 py-1 shrink-0">
        Feature Evolution Map
      </div>
      <div ref={containerRef} className="flex-1 min-h-0">
        <canvas ref={canvasRef} className="w-full h-full" />
      </div>
    </div>
  );
}
