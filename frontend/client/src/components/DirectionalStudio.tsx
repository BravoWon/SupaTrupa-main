import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Compass, SplitSquareHorizontal, Settings2 } from "lucide-react";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Label } from "@/components/ui/label";
import { OffsetWell } from "@/lib/offsetWells";

interface SurveyPoint {
  md: number;
  inc: number;
  azi: number;
  tvd: number;
  n_s: number;
  e_w: number;
  dls: number;
}

interface DirectionalStudioProps {
  surveys: SurveyPoint[];
  plan: SurveyPoint[];
  offsetWells?: OffsetWell[];
}

export function DirectionalStudio({ surveys, plan, offsetWells }: DirectionalStudioProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);

  const [isSideBySide, setIsSideBySide] = useState(false);

  const [actualColor, setActualColor] = useState("#ff0000");
  const [planColor, setPlanColor] = useState("#00ff00");
  const [lineWidth, setLineWidth] = useState(2);
  const [tunnelOpacity, setTunnelOpacity] = useState(0.1);

  useEffect(() => {
    if (!mountRef.current) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x09090b);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(75, mountRef.current.clientWidth / mountRef.current.clientHeight, 0.1, 10000);
    camera.position.set(500, 500, 500);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);

    renderer.setScissorTest(true);

    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controlsRef.current = controls;

    const gridHelper = new THREE.GridHelper(2000, 20, 0x444444, 0x222222);
    scene.add(gridHelper);
    const axesHelper = new THREE.AxesHelper(100);
    scene.add(axesHelper);

    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();

      if (isSideBySide) {
        const width = mountRef.current!.clientWidth;
        const height = mountRef.current!.clientHeight;

        renderer.setScissor(0, 0, width / 2, height);
        renderer.setViewport(0, 0, width / 2, height);

        renderer.render(scene, camera);

        renderer.setScissor(width / 2, 0, width / 2, height);
        renderer.setViewport(width / 2, 0, width / 2, height);
        renderer.render(scene, camera);
      } else {

        const width = mountRef.current!.clientWidth;
        const height = mountRef.current!.clientHeight;
        renderer.setScissor(0, 0, width, height);
        renderer.setViewport(0, 0, width, height);
        renderer.render(scene, camera);
      }
    };
    animate();

    return () => {
      mountRef.current?.removeChild(renderer.domElement);
      renderer.dispose();
    };
  }, [isSideBySide]);

  useEffect(() => {
    if (!sceneRef.current) return;
    const scene = sceneRef.current;

    scene.children = scene.children.filter(c => c.type === "GridHelper" || c.type === "AxesHelper");

    if (plan.length > 0) {
      const points = plan.map(p => new THREE.Vector3(p.e_w, -p.tvd, p.n_s));
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const material = new THREE.LineBasicMaterial({ color: new THREE.Color(planColor), linewidth: lineWidth });
      const line = new THREE.Line(geometry, material);
      scene.add(line);

      const curve = new THREE.CatmullRomCurve3(points);
      const tubeGeometry = new THREE.TubeGeometry(curve, points.length * 2, 20, 8, false);
      const tubeMaterial = new THREE.MeshBasicMaterial({
        color: new THREE.Color(planColor),
        opacity: tunnelOpacity,
        transparent: true,
        wireframe: true,
        side: THREE.DoubleSide
      });
      const tunnel = new THREE.Mesh(tubeGeometry, tubeMaterial);
      scene.add(tunnel);
    }

    if (surveys.length > 0) {
      const points = surveys.map(p => new THREE.Vector3(p.e_w, -p.tvd, p.n_s));
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const material = new THREE.LineBasicMaterial({ color: new THREE.Color(actualColor), linewidth: lineWidth });
      const line = new THREE.Line(geometry, material);
      scene.add(line);

      const lastPoint = points[points.length - 1];
      const bitGeo = new THREE.ConeGeometry(10, 20, 8);
      const bitMat = new THREE.MeshBasicMaterial({ color: 0xffff00 });
      const bit = new THREE.Mesh(bitGeo, bitMat);
      bit.position.copy(lastPoint);
      bit.rotation.x = Math.PI;
      scene.add(bit);

      if (offsetWells) {
        offsetWells.forEach(well => {

          let minDist = Infinity;
          well.points.forEach(p => {
            const dist = Math.sqrt(
              Math.pow(p.e_w - lastPoint.x, 2) +
              Math.pow(-p.tvd - lastPoint.y, 2) +
              Math.pow(p.n_s - lastPoint.z, 2)
            );
            if (dist < minDist) minDist = dist;
          });

          if (minDist < 50) {
            const warningGeo = new THREE.SphereGeometry(minDist, 16, 16);
            const warningMat = new THREE.MeshBasicMaterial({
              color: 0xff0000,
              wireframe: true,
              transparent: true,
              opacity: 0.3
            });
            const warningSphere = new THREE.Mesh(warningGeo, warningMat);
            warningSphere.position.copy(lastPoint);
            scene.add(warningSphere);
          }
        });
      }
    }

    if (offsetWells) {
      offsetWells.forEach(well => {
        const points = well.points.map(p => new THREE.Vector3(p.e_w, -p.tvd, p.n_s));
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({ color: 0x06b6d4, opacity: 0.5, transparent: true });
        const line = new THREE.Line(geometry, material);
        scene.add(line);
      });
    }

  }, [surveys, plan, offsetWells, actualColor, planColor, lineWidth, tunnelOpacity]);

  return (
    <Card className="h-full border-border bg-card/30 flex flex-col">
      <CardHeader className="pb-2 border-b border-border/50 flex flex-row items-center justify-between">
        <CardTitle className="text-sm font-mono flex items-center gap-2">
          <Compass className="w-4 h-4 text-chart-2" />
          DIRECTIONAL STUDIO
        </CardTitle>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="icon"
            className={`h-6 w-6 ${isSideBySide ? 'bg-primary/20' : ''}`}
            onClick={() => setIsSideBySide(!isSideBySide)}
            title="Toggle Side-by-Side View"
          >
            <SplitSquareHorizontal className="h-3 w-3" />
          </Button>

          <Popover>
            <PopoverTrigger asChild>
              <Button variant="outline" size="icon" className="h-6 w-6">
                <Settings2 className="h-3 w-3" />
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-64">
              <div className="space-y-4">
                <h4 className="font-medium leading-none">Visual Settings</h4>
                <div className="space-y-2">
                  <Label>Plan Color</Label>
                  <div className="flex gap-2">
                    {['#00ff00', '#00ffff', '#ff00ff'].map(c => (
                      <div
                        key={c}
                        className={`w-6 h-6 rounded-full cursor-pointer border ${planColor === c ? 'border-white' : 'border-transparent'}`}
                        style={{ backgroundColor: c }}
                        onClick={() => setPlanColor(c)}
                      />
                    ))}
                  </div>
                </div>
                <div className="space-y-2">
                  <Label>Actual Color</Label>
                  <div className="flex gap-2">
                    {['#ff0000', '#ff8800', '#ffff00'].map(c => (
                      <div
                        key={c}
                        className={`w-6 h-6 rounded-full cursor-pointer border ${actualColor === c ? 'border-white' : 'border-transparent'}`}
                        style={{ backgroundColor: c }}
                        onClick={() => setActualColor(c)}
                      />
                    ))}
                  </div>
                </div>
                <div className="space-y-2">
                  <Label>Line Width: {lineWidth}</Label>
                  <Slider
                    value={[lineWidth]}
                    min={1}
                    max={10}
                    step={1}
                    onValueChange={([v]) => setLineWidth(v)}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Tunnel Opacity: {tunnelOpacity.toFixed(1)}</Label>
                  <Slider
                    value={[tunnelOpacity]}
                    min={0}
                    max={1}
                    step={0.1}
                    onValueChange={([v]) => setTunnelOpacity(v)}
                  />
                </div>
              </div>
            </PopoverContent>
          </Popover>

          <Badge variant="outline" style={{ borderColor: planColor, color: planColor }}>PLAN</Badge>
          <Badge variant="outline" style={{ borderColor: actualColor, color: actualColor }}>ACTUAL</Badge>
        </div>
      </CardHeader>
      <div className="flex-1 relative" ref={mountRef}>
        {isSideBySide && (
          <div className="absolute inset-0 pointer-events-none flex">
            <div className="w-1/2 border-r border-white/10 flex items-start justify-center pt-4">
              <Badge variant="secondary" className="bg-black/50">TOP VIEW</Badge>
            </div>
            <div className="w-1/2 flex items-start justify-center pt-4">
              <Badge variant="secondary" className="bg-black/50">3D VIEW</Badge>
            </div>
          </div>
        )}
        <div className="absolute bottom-4 left-4 bg-black/60 backdrop-blur p-2 rounded border border-white/10 text-xs font-mono text-muted-foreground pointer-events-none">
          <div>TVD: {surveys[surveys.length-1]?.tvd.toFixed(1)} ft</div>
          <div>INC: {surveys[surveys.length-1]?.inc.toFixed(1)}°</div>
          <div>AZI: {surveys[surveys.length-1]?.azi.toFixed(1)}°</div>
        </div>
      </div>
    </Card>
  );
}
