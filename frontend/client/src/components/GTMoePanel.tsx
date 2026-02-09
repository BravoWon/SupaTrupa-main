import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { BrainCircuit, CheckCircle2, Zap } from 'lucide-react';
import type { OptimizationResult } from '@/lib/gtMoeOptimizer';

interface GTMoePanelProps {
  result: OptimizationResult | null;
  backendRegime?: string | null;
}

export function GTMoePanel({ result, backendRegime }: GTMoePanelProps) {
  if (!result) return (
    <Card className="h-full border-border bg-card/30 flex items-center justify-center">
      <div className="text-muted-foreground text-xs animate-pulse">INITIALIZING GT-MoE KERNEL...</div>
    </Card>
  );

  return (
    <Card className="h-full border-border bg-card/30 flex flex-col">
      <CardHeader className="pb-2 border-b border-border/50">
        <CardTitle className="text-sm font-mono flex items-center gap-2">
          <BrainCircuit className="w-4 h-4 text-chart-5" />
          GT-MoE BHA OPTIMIZER
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-4 space-y-4 font-mono text-sm flex-1 overflow-y-auto">

        <div className="flex items-center justify-between">
          <span className="text-muted-foreground text-xs">VIBRATION REGIME</span>
          <Badge variant="outline" className={`
            ${result.regime === 'Normal' ? 'text-green-500 border-green-500/50' :
              result.regime === 'Whirl' ? 'text-red-500 border-red-500/50 animate-pulse' :
              'text-yellow-500 border-yellow-500/50'}
          `}>
            {result.regime.toUpperCase()}
          </Badge>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div className="bg-black/40 p-2 rounded border border-border/50">
            <div className="text-[10px] text-muted-foreground">CLUSTERS (CV)</div>
            <div className="text-lg font-bold text-chart-1">{result.vibrationIndicators.clusters}</div>
          </div>
          <div className="bg-black/40 p-2 rounded border border-border/50">
            <div className="text-[10px] text-muted-foreground">CYCLES (CV)</div>
            <div className="text-lg font-bold text-chart-2">{result.vibrationIndicators.cycles}</div>
          </div>
        </div>

        <div className="space-y-2 pt-2 border-t border-border/50">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Zap className="w-3 h-3 text-chart-4" />
            <span>OPTIMIZATION STRATEGY</span>
          </div>

          <div className="bg-chart-5/10 border border-chart-5/20 p-3 rounded text-xs text-chart-5">
            &quot;{result.reasoning}&quot;
          </div>

          {result.regime !== 'Normal' && (
            <div className="space-y-1 mt-2">
              <div className="text-[10px] text-muted-foreground uppercase">Recommended Actions</div>
              {result.regime === 'Whirl' && (
                <div className="flex items-center gap-2 text-xs text-white">
                  <CheckCircle2 className="w-3 h-3 text-green-500" />
                  <span>Add Stabilizer (+1)</span>
                </div>
              )}
              {result.regime === 'Stick-Slip' && (
                <div className="flex items-center gap-2 text-xs text-white">
                  <CheckCircle2 className="w-3 h-3 text-green-500" />
                  <span>Reduce Motor Bend (-0.5&deg;)</span>
                </div>
              )}
              {result.regime === 'Bit Bounce' && (
                <div className="flex items-center gap-2 text-xs text-white">
                  <CheckCircle2 className="w-3 h-3 text-green-500" />
                  <span>Increase Flow Restrictor (+10%)</span>
                </div>
              )}
            </div>
          )}
        </div>

        {backendRegime && (
          <div className="flex items-center justify-between pt-2 border-t border-border/50">
            <span className="text-muted-foreground text-[10px]">TDA REGIME (BACKEND)</span>
            <Badge variant="outline" className="text-[10px] text-chart-3 border-chart-3/50">
              {backendRegime}
            </Badge>
          </div>
        )}

        <div className="mt-auto pt-2">
          <div className="flex justify-between text-[10px] text-muted-foreground mb-1">
            <span>VIBRATION CONFIDENCE</span>
            <span>{(result.confidence * 100).toFixed(1)}%</span>
          </div>
          <div className="h-1 w-full bg-secondary rounded-full overflow-hidden">
            <div
              className="h-full bg-chart-3 transition-all duration-500"
              style={{ width: `${result.confidence * 100}%` }}
            />
          </div>
        </div>

      </CardContent>
    </Card>
  );
}
