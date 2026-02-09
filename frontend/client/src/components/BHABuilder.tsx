import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Hammer, ArrowDown, Disc, Cylinder, Settings } from "lucide-react";

interface BHAComponent {
  id: string;
  type: "BIT" | "RSS" | "MUD_MOTOR" | "MWD" | "LWD" | "COLLAR" | "STABILIZER";
  name: string;
  length: number;
  od: number;
  id_inner: number;
  weight: number;
}

interface BHABuilderProps {
  config?: any;
  onChange?: (config: any) => void;
}

export function BHABuilder({ config, onChange }: BHABuilderProps) {
  void config; void onChange;
  const [components, setComponents] = useState<BHAComponent[]>([
    { id: "1", type: "BIT", name: "PDC 8.5in", length: 1.5, od: 8.5, id_inner: 0, weight: 150 },
    { id: "2", type: "RSS", name: "Push-the-Bit Rotary Steerable", length: 12.0, od: 6.75, id_inner: 2.5, weight: 120 },
    { id: "3", type: "MWD", name: "Pulser & Gamma", length: 30.0, od: 6.75, id_inner: 2.0, weight: 110 },
  ]);

  const addComponent = (type: BHAComponent["type"]) => {
    const newComp: BHAComponent = {
      id: Math.random().toString(36).substr(2, 9),
      type,
      name: `New ${type}`,
      length: 10.0,
      od: 6.5,
      id_inner: 2.5,
      weight: 100
    };
    setComponents([...components, newComp]);
  };

  const totalLength = components.reduce((acc, c) => acc + c.length, 0);
  const totalWeight = components.reduce((acc, c) => acc + (c.weight * c.length), 0);

  return (
    <Card className="h-full border-border bg-card/30">
      <CardHeader className="pb-2 border-b border-border/50 flex flex-row items-center justify-between">
        <CardTitle className="text-sm font-mono flex items-center gap-2">
          <Hammer className="w-4 h-4 text-chart-1" />
          BOTTOM HOLE ASSEMBLY (BHA)
        </CardTitle>
        <div className="text-xs font-mono text-muted-foreground">
          LEN: {totalLength.toFixed(1)} ft | WT: {(totalWeight / 1000).toFixed(1)} klbs
        </div>
      </CardHeader>
      <CardContent className="p-4 space-y-4">
        {}
        <div className="flex flex-col items-center gap-1 py-4 bg-black/20 rounded-lg border border-white/5 min-h-[300px] overflow-y-auto">
          {components.map((comp, idx) => (
            <div key={comp.id} className="relative group w-full flex justify-center">
              {}
              {idx > 0 && <div className="absolute -top-2 w-0.5 h-2 bg-muted-foreground/50" />}

              <div className="flex items-center gap-4 w-full max-w-[300px] px-4">
                {}
                <div className="w-8 flex justify-center">
                  {comp.type === "BIT" && <Disc className="w-6 h-6 text-chart-1 animate-spin-slow" />}
                  {comp.type === "RSS" && <Settings className="w-6 h-6 text-chart-2" />}
                  {comp.type === "MUD_MOTOR" && <Cylinder className="w-6 h-6 text-chart-3" />}
                  {(comp.type === "MWD" || comp.type === "LWD") && <ArrowDown className="w-6 h-6 text-chart-4" />}
                  {(comp.type === "COLLAR" || comp.type === "STABILIZER") && <div className="w-4 h-12 bg-muted rounded-sm" />}
                </div>

                {}
                <div className="flex-1 bg-card border border-border/50 p-2 rounded text-xs font-mono hover:border-primary transition-colors cursor-pointer">
                  <div className="font-bold text-foreground">{comp.name}</div>
                  <div className="text-muted-foreground flex justify-between mt-1">
                    <span>OD: {comp.od}"</span>
                    <span>L: {comp.length}'</span>
                  </div>
                </div>
              </div>
            </div>
          ))}

          {}
          <div className="mt-4 flex gap-2">
            <Button variant="outline" size="sm" onClick={() => addComponent("COLLAR")}>+ COLLAR</Button>
            <Button variant="outline" size="sm" onClick={() => addComponent("STABILIZER")}>+ STAB</Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
