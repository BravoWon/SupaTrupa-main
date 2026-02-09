import { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text } from '@react-three/drei';
import * as THREE from 'three';
import type { DrillingRecord } from '@/types';

interface AttractorManifoldProps {
  data: DrillingRecord[];
  currentDepth: number;
  regimeStatus: { regime: string; confidence: number; color: string } | null;
}

const GRID_SIZE = 50;
const SURFACE_SCALE = 6;

// Compute value surface from drilling records
// X = RPM (normalized), Y = WOB (normalized), Z = Value Potential
function computeSurface(data: DrillingRecord[]) {
  if (data.length < 3) {
    return { vertices: new Float32Array(0), colors: new Float32Array(0), rpmRange: [0, 1], wobRange: [0, 1] };
  }

  const rpms = data.map(d => d.rpm);
  const wobs = data.map(d => d.wob);
  const rops = data.map(d => d.rop);
  const torques = data.map(d => d.torque);

  const rpmMin = Math.min(...rpms);
  const rpmMax = Math.max(...rpms);
  const wobMin = Math.min(...wobs);
  const wobMax = Math.max(...wobs);
  const rpmRange = rpmMax - rpmMin || 1;
  const wobRange = wobMax - wobMin || 1;

  // Build 2D grid of value potential
  const vertices = new Float32Array(GRID_SIZE * GRID_SIZE * 3);
  const colors = new Float32Array(GRID_SIZE * GRID_SIZE * 3);

  // Pre-compute torque variance per record
  const avgTorque = torques.reduce((a, b) => a + b, 0) / torques.length;
  for (let iy = 0; iy < GRID_SIZE; iy++) {
    for (let ix = 0; ix < GRID_SIZE; ix++) {
      const idx = iy * GRID_SIZE + ix;
      const u = ix / (GRID_SIZE - 1); // 0..1
      const v = iy / (GRID_SIZE - 1); // 0..1

      const rpm = rpmMin + u * rpmRange;
      const wob = wobMin + v * wobRange;

      // Find nearby records and compute local value
      let weightedRop = 0;
      let weightedTorqueVar = 0;
      let totalWeight = 0;

      for (const rec of data) {
        const dr = (rec.rpm - rpm) / rpmRange;
        const dw = (rec.wob - wob) / wobRange;
        const dist2 = dr * dr + dw * dw;
        const w = Math.exp(-dist2 * 8); // Gaussian kernel
        weightedRop += w * rec.rop;
        weightedTorqueVar += w * (rec.torque - avgTorque) ** 2;
        totalWeight += w;
      }

      if (totalWeight > 0.001) {
        weightedRop /= totalWeight;
        weightedTorqueVar /= totalWeight;
      }

      // Value = ROP / (1 + torque_variance) — high ROP + low vibration = valley
      const value = weightedRop / (1 + weightedTorqueVar * 0.01);

      // Position: X = rpm, Y = value (inverted so valleys are low), Z = wob
      const x = (u - 0.5) * SURFACE_SCALE;
      const z = (v - 0.5) * SURFACE_SCALE;
      const y = -value * 0.03; // Scale and invert so good = valley

      vertices[idx * 3] = x;
      vertices[idx * 3 + 1] = y;
      vertices[idx * 3 + 2] = z;

      // Color: blue basins (efficient), red peaks (dysfunction)
      const normalizedValue = value / (Math.max(...rops) || 1);
      const hue = normalizedValue * 0.6; // 0 = red (dysfunction), 0.6 = blue (efficient)
      const color = new THREE.Color().setHSL(hue, 0.7, 0.45);
      colors[idx * 3] = color.r;
      colors[idx * 3 + 1] = color.g;
      colors[idx * 3 + 2] = color.b;
    }
  }

  return { vertices, colors, rpmRange: [rpmMin, rpmMax], wobRange: [wobMin, wobMax] };
}

function computeIndices(): Uint32Array {
  const indices: number[] = [];
  for (let iy = 0; iy < GRID_SIZE - 1; iy++) {
    for (let ix = 0; ix < GRID_SIZE - 1; ix++) {
      const a = iy * GRID_SIZE + ix;
      const b = a + 1;
      const c = a + GRID_SIZE;
      const d = c + 1;
      indices.push(a, c, b);
      indices.push(b, c, d);
    }
  }
  return new Uint32Array(indices);
}

function Surface({ data }: { data: DrillingRecord[] }) {
  const meshRef = useRef<THREE.Mesh>(null);

  const { vertices, colors } = useMemo(() => computeSurface(data), [data]);
  const indices = useMemo(() => computeIndices(), []);

  if (vertices.length === 0) return null;

  return (
    <mesh ref={meshRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="index"
          args={[indices, 1]}
        />
        <bufferAttribute
          attach="attributes-position"
          args={[vertices, 3]}
        />
        <bufferAttribute
          attach="attributes-color"
          args={[colors, 3]}
        />
      </bufferGeometry>
      <meshStandardMaterial
        vertexColors
        side={THREE.DoubleSide}
        roughness={0.7}
        metalness={0.1}
        transparent
        opacity={0.85}
        wireframe={false}
      />
    </mesh>
  );
}

function SurfaceWireframe({ data }: { data: DrillingRecord[] }) {
  const { vertices } = useMemo(() => computeSurface(data), [data]);
  const indices = useMemo(() => computeIndices(), []);

  if (vertices.length === 0) return null;

  return (
    <mesh>
      <bufferGeometry>
        <bufferAttribute attach="index" args={[indices, 1]} />
        <bufferAttribute attach="attributes-position" args={[vertices, 3]} />
      </bufferGeometry>
      <meshBasicMaterial color="#2dd4bf" wireframe opacity={0.15} transparent />
    </mesh>
  );
}

function CurrentStateDot({ data, currentDepth }: { data: DrillingRecord[]; currentDepth: number }) {
  const meshRef = useRef<THREE.Mesh>(null);

  const position = useMemo(() => {
    if (data.length < 3) return new THREE.Vector3(0, 0, 0);

    const rpms = data.map(d => d.rpm);
    const wobs = data.map(d => d.wob);
    const rpmMin = Math.min(...rpms);
    const rpmMax = Math.max(...rpms);
    const wobMin = Math.min(...wobs);
    const wobMax = Math.max(...wobs);
    const rpmRange = rpmMax - rpmMin || 1;
    const wobRange = wobMax - wobMin || 1;

    // Find closest record to current depth
    const current = data.reduce((best, r) =>
      Math.abs(r.depth - currentDepth) < Math.abs(best.depth - currentDepth) ? r : best
    );

    const u = (current.rpm - rpmMin) / rpmRange;
    const v = (current.wob - wobMin) / wobRange;

    // Compute local value at this point
    const avgTorque = data.reduce((a, d) => a + d.torque, 0) / data.length;
    let weightedRop = 0;
    let totalWeight = 0;
    for (const rec of data) {
      const dr = (rec.rpm - current.rpm) / rpmRange;
      const dw = (rec.wob - current.wob) / wobRange;
      const dist2 = dr * dr + dw * dw;
      const w = Math.exp(-dist2 * 8);
      weightedRop += w * rec.rop;
      totalWeight += w;
    }
    if (totalWeight > 0) weightedRop /= totalWeight;
    const value = weightedRop / (1 + (current.torque - avgTorque) ** 2 * 0.01);

    return new THREE.Vector3(
      (u - 0.5) * SURFACE_SCALE,
      -value * 0.03 + 0.1, // Slightly above surface
      (v - 0.5) * SURFACE_SCALE
    );
  }, [data, currentDepth]);

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.scale.setScalar(1 + Math.sin(state.clock.elapsedTime * 3) * 0.2);
    }
  });

  return (
    <mesh ref={meshRef} position={position}>
      <sphereGeometry args={[0.12, 16, 16]} />
      <meshStandardMaterial color="#ffffff" emissive="#ffffff" emissiveIntensity={0.5} />
    </mesh>
  );
}

function GeodesicPath({ data }: { data: DrillingRecord[] }) {
  const points = useMemo(() => {
    if (data.length < 10) return [];

    const rpms = data.map(d => d.rpm);
    const wobs = data.map(d => d.wob);
    const rpmMin = Math.min(...rpms);
    const rpmMax = Math.max(...rpms);
    const wobMin = Math.min(...wobs);
    const wobMax = Math.max(...wobs);
    const rpmRange = rpmMax - rpmMin || 1;
    const wobRange = wobMax - wobMin || 1;
    const avgTorque = data.reduce((a, d) => a + d.torque, 0) / data.length;

    // Use last N records as the "geodesic path" trace on the surface
    const recent = data.slice(-Math.min(30, data.length));
    return recent.map(rec => {
      const u = (rec.rpm - rpmMin) / rpmRange;
      const v = (rec.wob - wobMin) / wobRange;
      let weightedRop = 0;
      let totalWeight = 0;
      for (const r of data) {
        const dr = (r.rpm - rec.rpm) / rpmRange;
        const dw = (r.wob - rec.wob) / wobRange;
        const w = Math.exp(-(dr * dr + dw * dw) * 8);
        weightedRop += w * r.rop;
        totalWeight += w;
      }
      if (totalWeight > 0) weightedRop /= totalWeight;
      const value = weightedRop / (1 + (rec.torque - avgTorque) ** 2 * 0.01);
      return new THREE.Vector3(
        (u - 0.5) * SURFACE_SCALE,
        -value * 0.03 + 0.05,
        (v - 0.5) * SURFACE_SCALE
      );
    });
  }, [data]);

  if (points.length < 2) return null;

  const curve = new THREE.CatmullRomCurve3(points);
  const tubePoints = curve.getPoints(60);
  return (
    <line>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[new Float32Array(tubePoints.flatMap(p => [p.x, p.y, p.z])), 3]}
        />
      </bufferGeometry>
      <lineBasicMaterial color="#2dd4bf" linewidth={2} transparent opacity={0.8} />
    </line>
  );
}

function AxisLabels() {
  return (
    <>
      <Text position={[SURFACE_SCALE / 2 + 0.5, 0, 0]} fontSize={0.2} color="#6b7280" anchorX="left" font={undefined}>
        RPM
      </Text>
      <Text position={[0, 0, SURFACE_SCALE / 2 + 0.5]} fontSize={0.2} color="#6b7280" anchorX="center" font={undefined}>
        WOB
      </Text>
      <Text position={[-SURFACE_SCALE / 2 - 0.3, 0.5, 0]} fontSize={0.15} color="#6b7280" anchorX="right" font={undefined}>
        Value
      </Text>
    </>
  );
}

function Scene({ data, currentDepth }: { data: DrillingRecord[]; currentDepth: number }) {
  return (
    <>
      <ambientLight intensity={0.4} />
      <directionalLight position={[5, 8, 5]} intensity={0.6} />
      <directionalLight position={[-3, 5, -3]} intensity={0.3} />

      <Surface data={data} />
      <SurfaceWireframe data={data} />
      <CurrentStateDot data={data} currentDepth={currentDepth} />
      <GeodesicPath data={data} />
      <AxisLabels />

      <OrbitControls
        makeDefault
        enablePan
        enableZoom
        enableRotate
        autoRotate
        autoRotateSpeed={0.3}
        minDistance={3}
        maxDistance={15}
        target={[0, -0.5, 0]}
      />
    </>
  );
}

export function AttractorManifold({ data, currentDepth, regimeStatus }: AttractorManifoldProps) {
  return (
    <div className="h-full w-full relative">
      <Canvas
        camera={{ position: [4, 3, 4], fov: 50 }}
        gl={{ antialias: true, alpha: true }}
        style={{ background: 'transparent' }}
      >
        <Scene data={data} currentDepth={currentDepth} />
      </Canvas>
      {/* Overlay label */}
      <div className="absolute top-2 left-2 text-[10px] font-mono text-muted-foreground uppercase tracking-wider pointer-events-none">
        Attractor Manifold
      </div>
      {regimeStatus && (
        <div className="absolute bottom-2 left-2 text-[9px] font-mono text-muted-foreground pointer-events-none">
          <span className="text-primary">Efficient Drilling</span> = Valley &middot;{' '}
          <span className="text-destructive">Dysfunction</span> = Peak
        </div>
      )}
    </div>
  );
}
