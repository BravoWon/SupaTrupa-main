import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

interface Manifold3DProps {
  data: any[];
  axes: { [key: string]: boolean };
}

export function Manifold3D({ data, axes }: Manifold3DProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const pointsRef = useRef<THREE.Points | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);

  useEffect(() => {
    if (!mountRef.current) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x09090b);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(75, mountRef.current.clientWidth / mountRef.current.clientHeight, 0.1, 1000);
    camera.position.z = 50;
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controlsRef.current = controls;

    const gridHelper = new THREE.GridHelper(100, 100, 0x444444, 0x222222);
    scene.add(gridHelper);

    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      mountRef.current?.removeChild(renderer.domElement);
      renderer.dispose();
    };
  }, []);

  useEffect(() => {
    if (!sceneRef.current || !data.length) return;

    if (pointsRef.current) {
      sceneRef.current.remove(pointsRef.current);
      pointsRef.current.geometry.dispose();
      (pointsRef.current.material as THREE.Material).dispose();
    }

    const geometry = new THREE.BufferGeometry();
    const positions: number[] = [];
    const colors: number[] = [];

    const colorObj = new THREE.Color();

    data.forEach((point) => {

      const x = (point.depth % 1000) / 10 - 50;

      const y = (point.rop / 10) - 10;

      let z = 0;
      if (axes.torque) z = (point.torque / 1000) - 5;
      else if (axes.mse) z = (point.mse / 1000) - 5;

      positions.push(x, y, z);

      const surprise = point.surprise || 0;
      if (surprise > 2.0) {
        colorObj.setHex(0xff0000);
      } else {

        colorObj.setHSL(0.3 + (point.rop / 500) * 0.3, 1.0, 0.5);
      }
      colors.push(colorObj.r, colorObj.g, colorObj.b);
    });

    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: 0.5,
      vertexColors: true,
      transparent: true,
      opacity: 0.8,
      sizeAttenuation: true
    });

    const points = new THREE.Points(geometry, material);
    sceneRef.current.add(points);
    pointsRef.current = points;

  }, [data, axes]);

  return <div ref={mountRef} className="w-full h-full rounded-lg overflow-hidden border border-border/50" />;
}
