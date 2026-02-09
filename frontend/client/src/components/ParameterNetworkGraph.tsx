import { useEffect, useRef, useState, useCallback } from 'react';
import { cn } from '@/lib/utils';
import { getApiUrl } from '@/lib/api';
import { ForceSimulation, type SimNode, type SimEdge } from '@/lib/forceSimulation';
import type {
  DrillingRecord,
  LASCurveData,
  ParameterNode,
  NetworkGraph,
  NetworkStats,
} from '@/types';

// -- Category colors --------------------------------------------------------
const CATEGORY_COLORS: Record<string, string> = {
  mechanical: '#f97316',  // orange
  hydraulic: '#3b82f6',   // blue
  formation: '#a855f7',   // purple
  directional: '#06b6d4', // cyan
  vibration: '#ec4899',   // magenta
  performance: '#22c55e', // green
};

const HEALTH_RING_COLORS: Record<string, string> = {
  optimal: '#22c55e',
  caution: '#f59e0b',
  warning: '#ef4444',
  critical: '#d946ef',
};

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------
interface ParameterNetworkGraphProps {
  data: DrillingRecord[];
  lasCurveData: LASCurveData | null;
  selectedChannels: string[];
  onNodeClick: (nodeId: string) => void;
  onStatsUpdate: (stats: NetworkStats) => void;
  onGraphData?: (graph: NetworkGraph) => void;
  className?: string;
}

export function ParameterNetworkGraph({
  data,
  lasCurveData: _lasCurveData,
  selectedChannels,
  onNodeClick,
  onStatsUpdate,
  onGraphData,
  className,
}: ParameterNetworkGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const simRef = useRef<ForceSimulation | null>(null);
  const rafRef = useRef<number>(0);
  const nodesGRef = useRef<SVGGElement>(null);
  const edgesGRef = useRef<SVGGElement>(null);
  const labelsGRef = useRef<SVGGElement>(null);
  const [dims, setDims] = useState({ w: 800, h: 500 });
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [graphData, setGraphData] = useState<NetworkGraph | null>(null);
  const dragRef = useRef<{ id: string; offsetX: number; offsetY: number } | null>(null);
  const didDragRef = useRef(false);

  // -- Measure SVG container -----------------------------------------------
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const ro = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      if (width > 0 && height > 0) setDims({ w: width, h: height });
    });
    ro.observe(svg);
    return () => ro.disconnect();
  }, []);

  // -- Fetch network data from backend -------------------------------------
  useEffect(() => {
    if (data.length < 3 || selectedChannels.length < 2) return;

    const records = data.slice(-50).map((r) => ({
      wob: r.wob,
      rpm: r.rpm,
      rop: r.rop,
      torque: r.torque,
      spp: r.spp,
      hookload: r.hookload,
      depth: r.depth,
    }));

    const controller = new AbortController();
    fetch(getApiUrl('/api/v1/network/compute'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        records,
        channels: selectedChannels,
        window_size: 50,
        correlation_threshold: 0.3,
      }),
      signal: controller.signal,
    })
      .then((res) => {
        if (!res.ok) throw new Error(`Network compute failed: ${res.status}`);
        return res.json();
      })
      .then((g: NetworkGraph) => {
        setGraphData(g);
        onStatsUpdate({
          nodeCount: g.nodes.length,
          edgeCount: g.edges.length,
          strongCount: g.strong_count,
          anomalyCount: g.anomaly_count,
          systemHealth: g.system_health,
          computationTimeMs: g.computation_time_ms,
        });
        onGraphData?.(g);
      })
      .catch(() => {});

    return () => controller.abort();
  }, [data, selectedChannels]);

  // -- Build simulation when graphData changes -----------------------------
  useEffect(() => {
    if (!graphData || graphData.nodes.length === 0) return;

    const cx = dims.w / 2;
    const cy = dims.h / 2;

    const simNodes = ForceSimulation.initialLayout(
      graphData.nodes.map((n) => n.id),
      graphData.nodes.map((n) => n.category),
      cx,
      cy,
      Math.min(dims.w, dims.h) * 0.3,
    );

    // Set radius based on importance
    for (let i = 0; i < simNodes.length; i++) {
      simNodes[i].radius = 14 + graphData.nodes[i].importance * 18;
    }

    const simEdges: SimEdge[] = graphData.edges.map((e) => ({
      source: e.source,
      target: e.target,
      strength: Math.abs(e.pearson_r),
    }));

    const sim = new ForceSimulation(simNodes, simEdges, cx, cy);
    simRef.current = sim;

    // Animate
    cancelAnimationFrame(rafRef.current);
    const animate = () => {
      const running = sim.tick();
      renderSVG(sim, graphData);
      if (running) {
        rafRef.current = requestAnimationFrame(animate);
      }
    };
    rafRef.current = requestAnimationFrame(animate);

    return () => cancelAnimationFrame(rafRef.current);
  }, [graphData, dims]);

  // -- Direct SVG mutation (no React re-render per frame) -------------------
  const renderSVG = useCallback(
    (sim: ForceSimulation, gd: NetworkGraph) => {
      const edgesG = edgesGRef.current;
      const nodesG = nodesGRef.current;
      const labelsG = labelsGRef.current;
      if (!edgesG || !nodesG || !labelsG) return;

      const nodeMap = new Map<string, SimNode>();
      for (const n of sim.nodes) nodeMap.set(n.id, n);
      const gdMap = new Map<string, ParameterNode>();
      for (const n of gd.nodes) gdMap.set(n.id, n);

      // Edges
      const edgeEls = edgesG.children;
      for (let i = 0; i < gd.edges.length; i++) {
        const e = gd.edges[i];
        const s = nodeMap.get(e.source);
        const t = nodeMap.get(e.target);
        if (!s || !t) continue;
        const el = edgeEls[i] as SVGLineElement | undefined;
        if (!el) continue;
        el.setAttribute('x1', String(s.x));
        el.setAttribute('y1', String(s.y));
        el.setAttribute('x2', String(t.x));
        el.setAttribute('y2', String(t.y));
      }

      // Edge labels
      const edgeLabelG = labelsG.querySelector('#edge-labels');
      if (edgeLabelG) {
        const lblEls = edgeLabelG.children;
        for (let i = 0; i < gd.edges.length; i++) {
          const e = gd.edges[i];
          const s = nodeMap.get(e.source);
          const t = nodeMap.get(e.target);
          if (!s || !t) continue;
          const el = lblEls[i] as SVGTextElement | undefined;
          if (!el) continue;
          el.setAttribute('x', String((s.x + t.x) / 2));
          el.setAttribute('y', String((s.y + t.y) / 2 - 4));
        }
      }

      // Nodes
      const nodeEls = nodesG.children;
      for (let i = 0; i < sim.nodes.length; i++) {
        const n = sim.nodes[i];
        const el = nodeEls[i] as SVGGElement | undefined;
        if (!el) continue;
        el.setAttribute('transform', `translate(${n.x},${n.y})`);
      }
    },
    [],
  );

  // -- Pointer events for drag + click -------------------------------------
  const handlePointerDown = useCallback(
    (e: React.PointerEvent, nodeId: string) => {
      e.preventDefault();
      (e.target as Element).setPointerCapture(e.pointerId);
      const sim = simRef.current;
      if (!sim) return;
      const node = sim.nodes.find((n) => n.id === nodeId);
      if (!node) return;
      dragRef.current = { id: nodeId, offsetX: e.clientX - node.x, offsetY: e.clientY - node.y };
      didDragRef.current = false;
      sim.setFixed(nodeId, node.x, node.y);
      sim.wake();

      // Re-start animation
      cancelAnimationFrame(rafRef.current);
      const animate = () => {
        const running = sim.tick();
        if (graphData) renderSVG(sim, graphData);
        if (running) rafRef.current = requestAnimationFrame(animate);
      };
      rafRef.current = requestAnimationFrame(animate);
    },
    [graphData, renderSVG],
  );

  const handlePointerMove = useCallback(
    (e: React.PointerEvent) => {
      const d = dragRef.current;
      if (!d) return;
      didDragRef.current = true;
      const svg = svgRef.current;
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      simRef.current?.setFixed(d.id, x, y);
    },
    [],
  );

  const handlePointerUp = useCallback(
    (_e: React.PointerEvent) => {
      const d = dragRef.current;
      if (!d) return;
      simRef.current?.releaseFixed(d.id);
      if (!didDragRef.current) {
        onNodeClick(d.id);
      }
      dragRef.current = null;
    },
    [onNodeClick],
  );

  // -- Render ---------------------------------------------------------------
  if (!graphData || graphData.nodes.length === 0) {
    return (
      <svg
        ref={svgRef}
        className={cn('w-full h-full', className)}
      >
        <text
          x="50%"
          y="50%"
          textAnchor="middle"
          dominantBaseline="middle"
          fill="hsl(var(--muted-foreground))"
          fontSize="12"
          fontFamily="monospace"
        >
          Waiting for telemetry data...
        </text>
      </svg>
    );
  }

  const nodeMap = new Map<string, ParameterNode>();
  for (const n of graphData.nodes) nodeMap.set(n.id, n);

  return (
    <svg
      ref={svgRef}
      className={cn('w-full h-full select-none', className)}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
    >
      {/* Edges */}
      <g ref={edgesGRef}>
        {graphData.edges.map((e) => {
          const isHovered =
            hoveredNode !== null && (e.source === hoveredNode || e.target === hoveredNode);
          return (
            <line
              key={`${e.source}-${e.target}`}
              stroke={e.pearson_r >= 0 ? '#22c55e' : '#ef4444'}
              strokeWidth={1 + Math.abs(e.pearson_r) * 3}
              strokeOpacity={hoveredNode === null ? 0.35 : isHovered ? 0.8 : 0.08}
              x1={0}
              y1={0}
              x2={0}
              y2={0}
            />
          );
        })}
      </g>

      {/* Edge labels */}
      <g ref={labelsGRef}>
        <g id="edge-labels">
          {graphData.edges.map((e) => {
            const isHovered =
              hoveredNode !== null && (e.source === hoveredNode || e.target === hoveredNode);
            if (hoveredNode !== null && !isHovered) return null;
            if (Math.abs(e.pearson_r) < 0.5) return null;
            return (
              <text
                key={`lbl-${e.source}-${e.target}`}
                fill="hsl(var(--muted-foreground))"
                fontSize="9"
                fontFamily="monospace"
                textAnchor="middle"
                opacity={isHovered ? 1 : 0.6}
              >
                r={e.pearson_r.toFixed(2)}
              </text>
            );
          })}
        </g>
      </g>

      {/* Nodes */}
      <g ref={nodesGRef}>
        {graphData.nodes.map((n) => {
          const r = 14 + n.importance * 18;
          const catColor = CATEGORY_COLORS[n.category] ?? '#888';
          const healthColor = HEALTH_RING_COLORS[n.health] ?? '#888';
          const dimmed = hoveredNode !== null && hoveredNode !== n.id &&
            !graphData.edges.some(
              (e) =>
                (e.source === hoveredNode && e.target === n.id) ||
                (e.target === hoveredNode && e.source === n.id),
            );
          return (
            <g
              key={n.id}
              style={{ cursor: 'grab' }}
              opacity={dimmed ? 0.25 : 1}
              onPointerDown={(e) => handlePointerDown(e, n.id)}
              onPointerEnter={() => setHoveredNode(n.id)}
              onPointerLeave={() => setHoveredNode(null)}
            >
              {/* Health ring */}
              <circle
                r={r + 3}
                fill="none"
                stroke={healthColor}
                strokeWidth={2.5}
                strokeOpacity={0.8}
              />
              {/* Category fill */}
              <circle r={r} fill={catColor} fillOpacity={0.2} stroke={catColor} strokeWidth={1.5} />
              {/* Label */}
              <text
                y={-4}
                textAnchor="middle"
                fill={catColor}
                fontSize="10"
                fontWeight="bold"
                fontFamily="monospace"
                style={{ pointerEvents: 'none' }}
              >
                {n.id}
              </text>
              {/* Value */}
              <text
                y={8}
                textAnchor="middle"
                fill="hsl(var(--foreground))"
                fontSize="8"
                fontFamily="monospace"
                opacity={0.7}
                style={{ pointerEvents: 'none' }}
              >
                {n.current_value.toFixed(1)} {n.unit}
              </text>
              {/* Anomaly indicator */}
              {n.anomaly_flag && (
                <circle
                  cx={r - 2}
                  cy={-r + 2}
                  r={4}
                  fill="#ef4444"
                  stroke="#1a1a1a"
                  strokeWidth={1}
                />
              )}
            </g>
          );
        })}
      </g>

      {/* Category legend */}
      <g transform="translate(12, 16)">
        {Object.entries(CATEGORY_COLORS).map(([cat, color], i) => (
          <g key={cat} transform={`translate(0, ${i * 14})`}>
            <circle r={4} cx={4} cy={0} fill={color} fillOpacity={0.8} />
            <text
              x={12}
              y={3}
              fill="hsl(var(--muted-foreground))"
              fontSize="9"
              fontFamily="monospace"
              style={{ textTransform: 'capitalize' }}
            >
              {cat}
            </text>
          </g>
        ))}
      </g>
    </svg>
  );
}
