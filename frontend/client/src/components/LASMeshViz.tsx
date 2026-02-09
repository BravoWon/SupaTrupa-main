import { useMemo, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Text } from '@react-three/drei';
import * as THREE from 'three';
import type { LASCurveData } from '@/types';

type ColorMode = 'by-value' | 'by-curve' | 'by-regime';

interface LASMeshVizProps {
  data: LASCurveData;
  selectedCurves: string[];
  regimeColors?: string[];
}

const CURVE_COLORS = [
  '#3b82f6', '#22c55e', '#ef4444', '#eab308', '#a855f7',
  '#06b6d4', '#f97316', '#ec4899', '#14b8a6', '#8b5cf6',
];

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs((h / 60) % 2 - 1));
  const m = l - c / 2;
  let r = 0, g = 0, b = 0;
  if (h < 60) { r = c; g = x; }
  else if (h < 120) { r = x; g = c; }
  else if (h < 180) { g = c; b = x; }
  else if (h < 240) { g = x; b = c; }
  else if (h < 300) { r = x; b = c; }
  else { r = c; b = x; }
  return [r + m, g + m, b + m];
}

function WireMesh({
  data,
  selectedCurves,
  colorMode,
  showWireframe,
  showSolid,
  opacity,
}: {
  data: LASCurveData;
  selectedCurves: string[];
  colorMode: ColorMode;
  showWireframe: boolean;
  showSolid: boolean;
  opacity: number;
}) {
  const { geometry, wireGeometry } = useMemo(() => {
    const N = selectedCurves.length; // curves across X
    const indexVals = data.index_values;
    const M = indexVals.length; // depth points across Y

    if (N < 2 || M < 2) return { geometry: null, wireGeometry: null };

    // Normalize curve values per-curve (0..1)
    const normalizedValues: (number | null)[][] = [];
    const curveMinMax: { min: number; max: number }[] = [];

    for (const mnem of selectedCurves) {
      const curve = data.curves[mnem];
      if (!curve) {
        normalizedValues.push(Array(M).fill(null));
        curveMinMax.push({ min: 0, max: 1 });
        continue;
      }
      const vals = curve.values;
      const valid = vals.filter((v): v is number => v !== null && !isNaN(v));
      const min = valid.length > 0 ? Math.min(...valid) : 0;
      const max = valid.length > 0 ? Math.max(...valid) : 1;
      curveMinMax.push({ min, max });
      const range = max - min || 1;
      normalizedValues.push(vals.map(v => v !== null && !isNaN(v) ? (v - min) / range : null));
    }

    // Build geometry
    const positions = new Float32Array(N * M * 3);
    const colors = new Float32Array(N * M * 3);
    const indices: number[] = [];

    const depthMin = indexVals[0];
    const depthMax = indexVals[indexVals.length - 1];
    const depthRange = depthMax - depthMin || 1;

    for (let j = 0; j < M; j++) {
      for (let i = 0; i < N; i++) {
        const idx = j * N + i;

        // X: curve index spread
        const x = (i / (N - 1)) * 10 - 5;
        // Y: depth spread
        const y = ((indexVals[j] - depthMin) / depthRange) * 10 - 5;
        // Z: normalized value
        const normVal = normalizedValues[i][j];
        const z = normVal !== null ? normVal * 3 : 0;

        positions[idx * 3] = x;
        positions[idx * 3 + 1] = y;
        positions[idx * 3 + 2] = z;

        // Colors
        let r = 0.5, g = 0.5, b = 0.5;
        if (colorMode === 'by-value' && normVal !== null) {
          const hue = (1 - normVal) * 240; // blue(240) → red(0)
          [r, g, b] = hslToRgb(hue, 1, 0.5);
        } else if (colorMode === 'by-curve') {
          const hex = CURVE_COLORS[i % CURVE_COLORS.length];
          const c = new THREE.Color(hex);
          r = c.r; g = c.g; b = c.b;
        }

        colors[idx * 3] = r;
        colors[idx * 3 + 1] = g;
        colors[idx * 3 + 2] = b;
      }
    }

    // Build triangle indices
    for (let j = 0; j < M - 1; j++) {
      for (let i = 0; i < N - 1; i++) {
        const a = j * N + i;
        const b = j * N + i + 1;
        const c = (j + 1) * N + i;
        const d = (j + 1) * N + i + 1;
        indices.push(a, b, d);
        indices.push(a, d, c);
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.setIndex(indices);
    geo.computeVertexNormals();

    // Clone for wireframe
    const wireGeo = geo.clone();

    return { geometry: geo, wireGeometry: wireGeo };
  }, [data, selectedCurves, colorMode]);

  if (!geometry || !wireGeometry) return null;

  return (
    <group>
      {showSolid && (
        <mesh geometry={geometry}>
          <meshStandardMaterial
            vertexColors
            transparent
            opacity={opacity * 0.6}
            side={THREE.DoubleSide}
          />
        </mesh>
      )}
      {showWireframe && (
        <mesh geometry={wireGeometry}>
          <meshBasicMaterial
            vertexColors
            wireframe
            transparent
            opacity={opacity * 0.8}
          />
        </mesh>
      )}
    </group>
  );
}

export function LASMeshViz({ data, selectedCurves, regimeColors }: LASMeshVizProps) {
  void regimeColors;
  const [colorMode, setColorMode] = useState<ColorMode>('by-value');
  const [showWireframe, setShowWireframe] = useState(true);
  const [showSolid, setShowSolid] = useState(true);
  const [opacity, setOpacity] = useState(0.8);
  const [showControls, setShowControls] = useState(false);

  const effectiveCurves = selectedCurves.filter(c => c in data.curves);

  if (effectiveCurves.length < 2) {
    return (
      <div className="w-full h-full flex items-center justify-center text-muted-foreground text-sm font-mono">
        Select at least 2 curves for mesh visualization
      </div>
    );
  }

  return (
    <div className="w-full h-full relative bg-black/20 rounded-lg overflow-hidden border border-border/50">
      <Canvas camera={{ position: [8, 6, 8], fov: 45 }}>
        <color attach="background" args={['#0f172a']} />
        <fog attach="fog" args={['#0f172a', 10, 30]} />
        <ambientLight intensity={0.4} />
        <pointLight position={[10, 10, 10]} intensity={0.8} />
        <pointLight position={[-10, -5, 5]} intensity={0.3} />

        <WireMesh
          data={data}
          selectedCurves={effectiveCurves}
          colorMode={colorMode}
          showWireframe={showWireframe}
          showSolid={showSolid}
          opacity={opacity}
        />

        <gridHelper args={[20, 20, '#1e293b', '#1e293b']} position={[0, -6, 0]} />
        <OrbitControls />

        {/* Axis labels */}
        <Text position={[6, -5, 0]} fontSize={0.25} color="#94a3b8">
          Curves
        </Text>
        <Text position={[0, 6, 0]} fontSize={0.25} color="#94a3b8">
          Depth
        </Text>
        <Text position={[0, -5, 4]} fontSize={0.25} color="#94a3b8">
          Value
        </Text>
      </Canvas>

      {/* Controls overlay */}
      <div className="absolute top-2 right-2 z-10">
        <button
          onClick={() => setShowControls(!showControls)}
          className="bg-card/80 border border-border rounded p-1.5 hover:bg-card transition-colors"
        >
          <svg className="w-4 h-4 text-muted-foreground" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
            <circle cx="12" cy="12" r="3" />
          </svg>
        </button>
        {showControls && (
          <div className="mt-1 bg-card/90 border border-border rounded p-3 space-y-3 w-48 backdrop-blur">
            <div className="space-y-1">
              <span className="text-[10px] font-mono text-muted-foreground">COLOR MODE</span>
              {(['by-value', 'by-curve'] as const).map(mode => (
                <label key={mode} className="flex items-center gap-1.5 cursor-pointer">
                  <input
                    type="radio"
                    name="colorMode"
                    checked={colorMode === mode}
                    onChange={() => setColorMode(mode)}
                    className="accent-primary w-3 h-3"
                  />
                  <span className="text-[11px] font-mono capitalize">{mode.replace('-', ' ')}</span>
                </label>
              ))}
            </div>
            <div className="space-y-1">
              <span className="text-[10px] font-mono text-muted-foreground">DISPLAY</span>
              <label className="flex items-center gap-1.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showWireframe}
                  onChange={e => setShowWireframe(e.target.checked)}
                  className="accent-primary w-3 h-3"
                />
                <span className="text-[11px] font-mono">Wireframe</span>
              </label>
              <label className="flex items-center gap-1.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showSolid}
                  onChange={e => setShowSolid(e.target.checked)}
                  className="accent-primary w-3 h-3"
                />
                <span className="text-[11px] font-mono">Solid</span>
              </label>
            </div>
            <div className="space-y-1">
              <span className="text-[10px] font-mono text-muted-foreground">OPACITY</span>
              <input
                type="range"
                min="0.1"
                max="1"
                step="0.05"
                value={opacity}
                onChange={e => setOpacity(Number(e.target.value))}
                className="w-full h-1 accent-primary"
              />
            </div>
            <div className="text-[9px] text-muted-foreground font-mono">
              {effectiveCurves.length} curves x {data.index_values.length} pts
            </div>
          </div>
        )}
      </div>

      {/* Curve legend */}
      <div className="absolute bottom-2 left-2 z-10 bg-card/80 border border-border rounded p-2 max-w-[200px]">
        {effectiveCurves.map((mnem, i) => (
          <div key={mnem} className="flex items-center gap-1.5">
            <div
              className="w-2 h-2 rounded-full shrink-0"
              style={{ backgroundColor: CURVE_COLORS[i % CURVE_COLORS.length] }}
            />
            <span className="text-[10px] font-mono truncate">{mnem}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
