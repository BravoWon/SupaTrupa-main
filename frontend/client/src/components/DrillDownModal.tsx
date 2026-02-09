import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { DrillingRecord } from "@/types";
import { Activity, AlertTriangle, Database, Layers } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { ScrollArea } from "@/components/ui/scroll-area";

interface DrillDownModalProps {
  record: DrillingRecord | null;
  isOpen: boolean;
  onClose: () => void;
}

export default function DrillDownModal({ record, isOpen, onClose }: DrillDownModalProps) {
  if (!record) return null;

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-3xl bg-card border-border font-mono">
        <DialogHeader className="border-b border-border pb-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className={cn(
                "w-3 h-3 rounded-full animate-pulse",
                record.surprise > 2.0 ? "bg-chart-2" : "bg-chart-3"
              )} />
              <DialogTitle className="text-xl font-bold tracking-tight">
                DATA POINT INSPECTION
              </DialogTitle>
            </div>
            <Badge variant="outline" className="font-mono">
              ID: {record.id}
            </Badge>
          </div>
          <DialogDescription className="text-muted-foreground mt-1">
            Detailed analysis of engine state and raw sensor values at depth {record.depth.toFixed(2)} ft
          </DialogDescription>
        </DialogHeader>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 py-4">
          {}
          <div className="space-y-6">
            <div>
              <h3 className="text-sm font-bold text-chart-1 mb-3 flex items-center gap-2">
                <Activity className="w-4 h-4" />
                ENGINE STATE
              </h3>
              <div className="grid grid-cols-2 gap-4 bg-muted/30 p-4 rounded-lg border border-border/50">
                <div>
                  <div className="text-xs text-muted-foreground mb-1">SURPRISE METRIC</div>
                  <div className={cn(
                    "text-2xl font-bold",
                    record.surprise > 2.0 ? "text-chart-2" : "text-foreground"
                  )}>
                    {record.surprise.toFixed(4)}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground mb-1">CONFIDENCE</div>
                  <div className="text-2xl font-bold text-chart-3">
                    {(record.confidence * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="col-span-2 pt-2 border-t border-border/30">
                  <div className="text-xs text-muted-foreground mb-1">ACTIVITY CLASSIFICATION</div>
                  <Badge variant="outline" className={cn(
                    "w-full justify-center py-1.5 text-xs uppercase tracking-widest",
                    record.activity_type === 'spike' ? "bg-chart-2/20 text-chart-2 border-chart-2/50" :
                    record.activity_type === 'subthreshold' ? "bg-chart-4/20 text-chart-4 border-chart-4/50" :
                    "bg-muted text-muted-foreground"
                  )}>
                    {record.activity_type}
                  </Badge>
                </div>
              </div>
            </div>

            <div>
              <h3 className="text-sm font-bold text-chart-5 mb-3 flex items-center gap-2">
                <Layers className="w-4 h-4" />
                STATE VECTOR (μ)
              </h3>
              <ScrollArea className="h-[120px] w-full rounded-md border border-border/50 bg-muted/30 p-4">
                <div className="space-y-2 font-mono text-xs">
                  {record.mu.map((val, idx) => (
                    <div key={idx} className="flex justify-between items-center">
                      <span className="text-muted-foreground">DIMENSION {idx + 1}</span>
                      <span className="text-foreground font-bold">{val.toFixed(6)}</span>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </div>
          </div>

          {}
          <div className="space-y-6">
            <div>
              <h3 className="text-sm font-bold text-chart-4 mb-3 flex items-center gap-2">
                <Database className="w-4 h-4" />
                RAW SENSOR VALUES
              </h3>
              <div className="space-y-3 bg-muted/30 p-4 rounded-lg border border-border/50">
                <div className="flex justify-between items-center pb-2 border-b border-border/30">
                  <span className="text-sm text-muted-foreground">DEPTH</span>
                  <span className="text-lg font-bold text-foreground">{record.depth.toFixed(2)} <span className="text-xs text-muted-foreground">ft</span></span>
                </div>
                <div className="flex justify-between items-center pb-2 border-b border-border/30">
                  <span className="text-sm text-muted-foreground">ROP</span>
                  <span className="text-lg font-bold text-chart-1">{record.rop.toFixed(2)} <span className="text-xs text-muted-foreground">ft/hr</span></span>
                </div>
                <div className="flex justify-between items-center pb-2 border-b border-border/30">
                  <span className="text-sm text-muted-foreground">WOB</span>
                  <span className="text-lg font-bold text-chart-5">{record.wob.toFixed(0)} <span className="text-xs text-muted-foreground">klbf</span></span>
                </div>
                <div className="flex justify-between items-center pb-2 border-b border-border/30">
                  <span className="text-sm text-muted-foreground">HOOKLOAD</span>
                  <span className="text-lg font-bold text-chart-3">{record.hookload.toFixed(0)} <span className="text-xs text-muted-foreground">klbf</span></span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">SPP</span>
                  <span className="text-lg font-bold text-chart-4">{record.spp.toFixed(0)} <span className="text-xs text-muted-foreground">psi</span></span>
                </div>
              </div>
            </div>

            <div className="bg-chart-2/10 border border-chart-2/30 rounded-lg p-4">
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-chart-2 mt-0.5" />
                <div>
                  <h4 className="text-sm font-bold text-chart-2 mb-1">ANOMALY ANALYSIS</h4>
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    {record.surprise > 2.0
                      ? "High surprise detected. The observed sensor values deviate significantly from the engine's internal prediction model. This suggests a potential drilling dysfunction or formation change."
                      : "System operating within expected parameters. Low surprise indicates the internal model is accurately predicting drilling dynamics."
                    }
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
