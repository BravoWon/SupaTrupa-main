import { useRef, useEffect, useMemo } from 'react';
import { cn } from '@/lib/utils';
import type { MetricFieldPoint } from '@/types';

interface CurvatureFieldProps {
  points: MetricFieldPoint[];
  tValues: number[];
  dValues: number[];
  resolution: number;
  className?: string;
}

/**
 * Canvas heatmap of Ricci scalar curvature over (time, depth) parameter space.
 *
 * Color mapping:
 * - Blue (negative R): diverging geometry, parameters separating
 * - Green (R ~ 0): flat/Euclidean, stable region
 * - Yellow/Red (positive R): converging geometry, parameters attracting
 *
 * ROP overlay: contour-like lines showing effective ROP at each grid point.
 */
export function CurvatureField({
  points,
  tValues,
  dValues,
  resolution: _res,
  className,
}: CurvatureFieldProps) {
  void _res;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const { grid, maxAbsR } = useMemo(() => {
    if (points.length === 0) {
      return { grid: [] as number[][], maxAbsR: 1 };
    }
    // Build 2D grid of Ricci scalars: grid[row=d][col=t]
    const nT = tValues.length;
    const nD = dValues.length;
    const grid: number[][] = Array.from({ length: nD }, () => new Array(nT).fill(0));
    const ropGrid: number[][] = Array.from({ length: nD }, () => new Array(nT).fill(0));
    let maxAbsR = 0;

    for (const p of points) {
      const tIdx = tValues.findIndex(tv => Math.abs(tv - p.t) < 1e-6);
      const dIdx = dValues.findIndex(dv => Math.abs(dv - p.d) < 1e-6);
      if (tIdx >= 0 && dIdx >= 0) {
        grid[dIdx][tIdx] = p.ricci_scalar;
        ropGrid[dIdx][tIdx] = p.rop;
        if (Math.abs(p.ricci_scalar) > maxAbsR) maxAbsR = Math.abs(p.ricci_scalar);
      }
    }

    return { grid, maxAbsR: maxAbsR || 1 };
  }, [points, tValues, dValues]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || grid.length === 0) return;

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

    const labelW = 52;
    const labelH = 20;
    const padTop = 20;
    const padRight = 60; // For colorbar
    const plotW = w - labelW - padRight;
    const plotH = h - padTop - labelH;

    const nT = tValues.length;
    const nD = dValues.length;
    const cellW = plotW / nT;
    const cellH = plotH / nD;

    // Clear
    ctx.fillStyle = 'hsl(222, 30%, 6%)';
    ctx.fillRect(0, 0, w, h);

    // Draw heatmap cells
    for (let row = 0; row < nD; row++) {
      for (let col = 0; col < nT; col++) {
        const R = grid[row][col];
        const norm = R / maxAbsR; // [-1, 1]

        let r: number, g: number, b: number;
        if (norm < -0.01) {
          // Negative curvature: blue tones (diverging)
          const t = Math.min(1, Math.abs(norm));
          r = Math.round(20 + t * 30);
          g = Math.round(40 + t * 60);
          b = Math.round(80 + t * 175);
        } else if (norm > 0.01) {
          // Positive curvature: yellow to red (converging)
          const t = Math.min(1, norm);
          if (t < 0.5) {
            const s = t * 2;
            r = Math.round(40 + s * 200);
            g = Math.round(120 + s * 100);
            b = Math.round(40 - s * 20);
          } else {
            const s = (t - 0.5) * 2;
            r = Math.round(240 + s * 15);
            g = Math.round(220 - s * 180);
            b = Math.round(20 - s * 10);
          }
        } else {
          // Near zero: dark green (flat)
          r = 25;
          g = 60;
          b = 40;
        }

        const x = labelW + col * cellW;
        const y = padTop + (nD - 1 - row) * cellH; // Invert Y so depth increases downward
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(x, y, cellW + 0.5, cellH + 0.5);
      }
    }

    // Axis labels
    ctx.fillStyle = 'hsl(210, 15%, 55%)';
    ctx.font = '8px monospace';

    // Y-axis (depth)
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const dStep = Math.max(1, Math.floor(nD / 5));
    for (let i = 0; i < nD; i += dStep) {
      const y = padTop + (nD - 1 - i) * cellH + cellH / 2;
      ctx.fillText(`${dValues[i].toFixed(0)}`, labelW - 4, y);
    }

    // X-axis (time)
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const tStep = Math.max(1, Math.floor(nT / 5));
    for (let j = 0; j < nT; j += tStep) {
      const x = labelW + j * cellW + cellW / 2;
      ctx.fillText(`${tValues[j].toFixed(0)}`, x, padTop + plotH + 2);
    }

    // Title labels
    ctx.fillStyle = 'hsl(210, 15%, 45%)';
    ctx.font = '9px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Time (s)', labelW + plotW / 2, h - 2);

    ctx.save();
    ctx.translate(8, padTop + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Depth (ft)', 0, 0);
    ctx.restore();

    // Colorbar
    const cbX = w - padRight + 10;
    const cbW = 12;
    const cbH = plotH;
    for (let i = 0; i < cbH; i++) {
      const norm = 1 - (i / cbH) * 2; // top = +1, bottom = -1
      let cr: number, cg: number, cb: number;
      if (norm < -0.01) {
        const t = Math.min(1, Math.abs(norm));
        cr = Math.round(20 + t * 30);
        cg = Math.round(40 + t * 60);
        cb = Math.round(80 + t * 175);
      } else if (norm > 0.01) {
        const t = Math.min(1, norm);
        if (t < 0.5) {
          const s = t * 2;
          cr = Math.round(40 + s * 200);
          cg = Math.round(120 + s * 100);
          cb = Math.round(40 - s * 20);
        } else {
          const s = (t - 0.5) * 2;
          cr = Math.round(240 + s * 15);
          cg = Math.round(220 - s * 180);
          cb = Math.round(20 - s * 10);
        }
      } else {
        cr = 25;
        cg = 60;
        cb = 40;
      }
      ctx.fillStyle = `rgb(${cr},${cg},${cb})`;
      ctx.fillRect(cbX, padTop + i, cbW, 1);
    }

    // Colorbar labels
    ctx.fillStyle = 'hsl(210, 15%, 55%)';
    ctx.font = '7px monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(`+${maxAbsR.toFixed(2)}`, cbX + cbW + 3, padTop + 4);
    ctx.fillText('0', cbX + cbW + 3, padTop + cbH / 2);
    ctx.fillText(`-${maxAbsR.toFixed(2)}`, cbX + cbW + 3, padTop + cbH - 4);

    ctx.font = '8px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('R', cbX + cbW / 2, padTop - 8);
  }, [grid, maxAbsR, tValues, dValues]);

  if (points.length === 0) {
    return (
      <div className={cn('flex items-center justify-center text-muted-foreground text-[10px] font-mono h-full', className)}>
        Computing curvature field...
      </div>
    );
  }

  return (
    <div className={cn('w-full h-full flex flex-col', className)}>
      <div className="text-[9px] font-mono text-muted-foreground uppercase tracking-wider px-2 py-1 shrink-0 flex items-center justify-between">
        <span>Parameter Sensitivity Field</span>
        <span className="text-[8px]">
          <span className="text-blue-400">- diverging</span>
          {' '}
          <span className="text-green-400">stable</span>
          {' '}
          <span className="text-yellow-400">+ converging</span>
        </span>
      </div>
      <div ref={containerRef} className="flex-1 min-h-0">
        <canvas ref={canvasRef} className="w-full h-full" />
      </div>
    </div>
  );
}
