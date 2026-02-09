import { useRef, useEffect, useMemo, useState } from 'react';
import { cn } from '@/lib/utils';
import type { ShadowEmbedResponse } from '@/types';

interface DelayEmbeddingProps {
  embedding: ShadowEmbedResponse | null;
  className?: string;
}

/**
 * Canvas-based 3D scatter plot of delay-coordinate reconstruction.
 * Isometric projection with slow auto-rotation and time-based coloring.
 */
export function DelayEmbedding({ embedding, className }: DelayEmbeddingProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const angleRef = useRef(0);
  const rafRef = useRef(0);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  void setHoveredIdx;

  // Normalize point cloud to [-1,1] range
  const normalized = useMemo(() => {
    if (!embedding || embedding.point_cloud.length === 0) return null;
    const pc = embedding.point_cloud;
    const dim = pc[0].length;

    // Find bounds per dimension
    const mins = new Array(dim).fill(Infinity);
    const maxs = new Array(dim).fill(-Infinity);
    for (const pt of pc) {
      for (let d = 0; d < dim; d++) {
        mins[d] = Math.min(mins[d], pt[d]);
        maxs[d] = Math.max(maxs[d], pt[d]);
      }
    }

    const ranges = mins.map((mn, d) => maxs[d] - mn || 1);
    return pc.map(pt =>
      pt.map((v, d) => ((v - mins[d]) / ranges[d]) * 2 - 1)
    );
  }, [embedding]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || !normalized) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const draw = () => {
      const w = container.clientWidth;
      const h = container.clientHeight;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      ctx.fillStyle = 'hsl(220, 20%, 4%)';
      ctx.fillRect(0, 0, w, h);

      const cx = w / 2;
      const cy = h / 2;
      const scale = Math.min(w, h) * 0.32;
      const angle = angleRef.current;

      const cosA = Math.cos(angle);
      const sinA = Math.sin(angle);
      const tilt = 0.6; // Isometric tilt

      // Project 3D -> 2D with rotation around Y axis
      const project = (pt: number[]): [number, number, number] => {
        const x = pt[0] || 0;
        const y = pt[1] || 0;
        const z = pt[2] || 0;

        const rx = x * cosA - z * sinA;
        const rz = x * sinA + z * cosA;
        const ry = y;

        const px = cx + rx * scale;
        const py = cy - ry * scale * tilt + rz * scale * 0.3;
        return [px, py, rz]; // rz for depth sorting
      };

      // Project and sort by depth
      const projected = normalized.map((pt, i) => {
        const [px, py, depth] = project(pt);
        return { px, py, depth, idx: i };
      });
      projected.sort((a, b) => a.depth - b.depth);

      const n = normalized.length;

      // Draw connecting lines (trajectory)
      ctx.beginPath();
      ctx.strokeStyle = 'rgba(20, 184, 166, 0.08)';
      ctx.lineWidth = 0.5;
      // Sort by original index for lines
      const byIdx = [...projected].sort((a, b) => a.idx - b.idx);
      for (let i = 1; i < byIdx.length; i++) {
        ctx.moveTo(byIdx[i - 1].px, byIdx[i - 1].py);
        ctx.lineTo(byIdx[i].px, byIdx[i].py);
      }
      ctx.stroke();

      // Draw points (depth-sorted)
      for (const { px, py, idx } of projected) {
        const t = idx / (n - 1 || 1);
        // Time gradient: teal → amber → red
        const r = Math.round(20 + t * 225);
        const g = Math.round(184 - t * 100);
        const b2 = Math.round(166 - t * 140);
        const alpha = 0.5 + (idx === hoveredIdx ? 0.5 : 0);
        const radius = idx === hoveredIdx ? 4 : 2;

        ctx.beginPath();
        ctx.arc(px, py, radius, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${r}, ${g}, ${b2}, ${alpha})`;
        ctx.fill();
      }

      // Draw axes
      // axis length for labels
      const axes = [
        { label: `x(t)`, dir: [1, 0, 0], color: '#14b8a6' },
        { label: `x(t+${embedding?.delay_lag ?? 1})`, dir: [0, 1, 0], color: '#f59e0b' },
        { label: `x(t+${(embedding?.delay_lag ?? 1) * 2})`, dir: [0, 0, 1], color: '#ef4444' },
      ];

      const origin = project([0, 0, 0]);
      for (const axis of axes) {
        const end = project(axis.dir.map(v => v * 0.35));
        ctx.beginPath();
        ctx.moveTo(origin[0], origin[1]);
        ctx.lineTo(end[0], end[1]);
        ctx.strokeStyle = axis.color;
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.fillStyle = axis.color;
        ctx.font = '9px monospace';
        ctx.fillText(axis.label, end[0] + 3, end[1] - 3);
      }

      // Auto-rotate
      angleRef.current += 0.003;
      rafRef.current = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(rafRef.current);
  }, [normalized, hoveredIdx, embedding]);

  if (!embedding) {
    return (
      <div className={cn('flex items-center justify-center text-muted-foreground text-[10px] font-mono h-full', className)}>
        Awaiting shadow tensor embedding...
      </div>
    );
  }

  return (
    <div ref={containerRef} className={cn('w-full h-full flex flex-col', className)}>
      <div className="text-[9px] font-mono text-muted-foreground uppercase tracking-wider px-2 py-1 shrink-0 flex justify-between">
        <span>Hidden Dynamics</span>
        <span className="text-primary">
          dim={embedding.embedding_dim} lag={embedding.delay_lag} n={embedding.n_points}
        </span>
      </div>
      <div className="flex-1 min-h-0 relative">
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full"
        />
      </div>
      {/* Proxy summary */}
      <div className="px-2 py-1 text-[8px] font-mono text-muted-foreground flex gap-3 shrink-0 border-t border-border/30">
        <span>
          BBW: {embedding.metric_proxy.map(v => v.toFixed(2)).join(', ')}
        </span>
        <span>
          RSI: {embedding.tangent_proxy.filter((_, i) => i % 2 === 0).map(v => v.toFixed(1)).join(', ')}
        </span>
      </div>
    </div>
  );
}
