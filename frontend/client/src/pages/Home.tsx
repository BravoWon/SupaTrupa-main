import { useEffect, useState, useMemo, useCallback } from 'react';
import { FileDown, Upload } from 'lucide-react';
import { Button } from '@/components/ui/button';
import DrillDownModal from '@/components/DrillDownModal';
import { DirectionalStudio } from '@/components/DirectionalStudio';
import { BHABuilder } from '@/components/BHABuilder';
import { SurveyImport } from '@/components/SurveyImport';
import { LASLoader } from '@/components/LASLoader';
import { LogTrackViewer } from '@/components/LogTrackViewer';
import { LASMeshViz } from '@/components/LASMeshViz';
import { GTMoePanel } from '@/components/GTMoePanel';
import { AttractorManifold } from '@/components/AttractorManifold';
import { PersistenceBarcode } from '@/components/PersistenceBarcode';
import { TrustGauge } from '@/components/TrustGauge';
import { CTSPipelineBar, type CTSStage } from '@/components/CTSPipelineBar';
import { KPICards } from '@/components/KPICards';
import { ParameterNetworkGraph } from '@/components/ParameterNetworkGraph';
import { ParameterDetailCard } from '@/components/ParameterDetailCard';
import { NetworkStatsBar } from '@/components/NetworkStatsBar';
import { ChannelSelector } from '@/components/ChannelSelector';
import { BettiTimeline } from '@/components/BettiTimeline';
import { TopologicalHeatmap } from '@/components/TopologicalHeatmap';
import { ChangeDetector } from '@/components/ChangeDetector';
import { CurvatureField } from '@/components/CurvatureField';
import { GeodesicOverlay } from '@/components/GeodesicOverlay';
import { RegimeFingerprint } from '@/components/RegimeFingerprint';
import { AttributionBars } from '@/components/AttributionBars';
import { RegimeCompare } from '@/components/RegimeCompare';
import { DelayEmbedding } from '@/components/DelayEmbedding';
import { LyapunovIndicator } from '@/components/LyapunovIndicator';
import { TopologyForecast } from '@/components/TopologyForecast';
import { TransitionRadar } from '@/components/TransitionRadar';
import { AdvisoryPanel } from '@/components/AdvisoryPanel';
import { GeodesicNavigator } from '@/components/GeodesicNavigator';
import { FieldAtlas } from '@/components/FieldAtlas';
import { WellCompare } from '@/components/WellCompare';
import { LASAnalyzer } from '@/components/LASAnalyzer';
import { RegimeStrip } from '@/components/RegimeStrip';
import { MasterDashboard } from '@/components/MasterDashboard';
import { usePersistentState } from '@/hooks/usePersistentState';
import { DrillingRecord } from '@/types';
import type {
  LASFileState, LASCurveData, NetworkStats, NetworkGraph,
  WindowedSignature, ChangeDetectResult,
  MetricFieldPoint, GeodesicResponse,
  FingerprintResponse, AttributionResponse, RegimeCompareResponse,
  ShadowEmbedResponse, AttractorAnalysis,
  ForecastResponse, TransitionProbResponse,
  AdvisoryResponse, RiskAssessmentResponse,
  FieldAtlasResponse, FieldCompareResponse,
  AnalyzerHistoryEntry, DashboardSummary,
  LASAnalyzeWindowResponse,
} from '@/types';
import { cn } from '@/lib/utils';
import { OffsetWellService, OffsetWell } from '@/lib/offsetWells';
import { getApiUrl } from '@/lib/api';
import { GTMoeOptimizer, ANALYSIS_WINDOW, type OptimizationResult } from '@/lib/gtMoeOptimizer';
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger
} from '@/components/ui/dialog';

type ViewTab = 'dashboard' | 'cts' | 'wellpath' | 'wiremesh' | 'network' | 'geometry' | 'fingerprint' | 'shadow' | 'forecast' | 'advisory' | 'field' | 'analyzer';

/** Full backend drilling/ingest response (single source of truth). */
interface BackendRegimeResponse {
  regime: string;
  confidence: number;
  color: string;
  betti_0: number;
  betti_1: number;
  recommendation: string;
  persistence_diagram?: {
    h0: { birth: number; death: number }[];
    h1: { birth: number; death: number }[];
    filtration_range: [number, number];
  } | null;
}

export default function Home() {
  const [data, setData] = useState<DrillingRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [currentDepth, setCurrentDepth] = useState<number>(0);
  const [selectedRecord, setSelectedRecord] = useState<DrillingRecord | null>(null);
  void setSelectedRecord;
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<ViewTab>('dashboard');

  const [bhaConfig, setBhaConfig] = usePersistentState<any>('bha-config', {
    bitType: 'PDC',
    motorBendAngle: 1.5,
    rssType: 'push-the-bit' as const,
    flowRestrictor: 50,
    stabilizers: 2
  });

  const [surveys, setSurveys] = useState<any[]>([]);
  const [plan, setPlan] = useState<any[]>([]);

  // LAS state
  const [lasFileState, setLasFileState] = useState<LASFileState | null>(null);
  const [lasCurveData, setLasCurveData] = useState<LASCurveData | null>(null);
  const [lasSelectedCurves, setLasSelectedCurves] = useState<string[]>([]);
  const [lasOverlayData, setLasOverlayData] = useState<DrillingRecord[] | null>(null);
  void lasOverlayData;

  // Cycle 2: Topological Time Machine state
  const [windowedSignatures, setWindowedSignatures] = useState<WindowedSignature[]>([]);
  const [changeResult, setChangeResult] = useState<ChangeDetectResult | null>(null);

  // Cycle 3: Manifold Geometry state
  const [curvaturePoints, setCurvaturePoints] = useState<MetricFieldPoint[]>([]);
  const [curvatureTValues, setCurvatureTValues] = useState<number[]>([]);
  const [curvatureDValues, setCurvatureDValues] = useState<number[]>([]);
  const [curvatureResolution, setCurvatureResolution] = useState(0);
  const [geodesicResult, setGeodesicResult] = useState<GeodesicResponse | null>(null);

  // Cycle 4: Persistence Fingerprinting state
  const [fingerprintData, setFingerprintData] = useState<FingerprintResponse | null>(null);
  const [attributionData, setAttributionData] = useState<AttributionResponse | null>(null);
  const [regimeCompareData, setRegimeCompareData] = useState<RegimeCompareResponse | null>(null);

  // Cycle 5: Predictive Topology state
  const [forecastData, setForecastData] = useState<ForecastResponse | null>(null);
  const [transitionData, setTransitionData] = useState<TransitionProbResponse | null>(null);

  // Cycle 6: Shadow Tensor state
  const [shadowEmbedding, setShadowEmbedding] = useState<ShadowEmbedResponse | null>(null);
  const [attractorAnalysis, setAttractorAnalysis] = useState<AttractorAnalysis | null>(null);

  // Cycle 7: Autonomous Advisory state
  const [advisoryData, setAdvisoryData] = useState<AdvisoryResponse | null>(null);
  const [riskData, setRiskData] = useState<RiskAssessmentResponse | null>(null);

  // Cycle 8: Field-Level Intelligence state
  const [fieldAtlasData, setFieldAtlasData] = useState<FieldAtlasResponse | null>(null);
  const [fieldCompareData, setFieldCompareData] = useState<FieldCompareResponse | null>(null);
  const [fieldSelectedWells, setFieldSelectedWells] = useState<[string | null, string | null]>([null, null]);
  const [fieldRegistered, setFieldRegistered] = useState(false);

  // Cycle 9: LAS Analyzer state
  const [lasAnalyzerHistory, setLasAnalyzerHistory] = useState<AnalyzerHistoryEntry[]>([]);
  const [lasWindowRecords, setLasWindowRecords] = useState<Record<string, number>[]>([]);
  const [lasWindowRegime, setLasWindowRegime] = useState<LASAnalyzeWindowResponse['regime']>(null);
  const [lasWindowSigs, setLasWindowSigs] = useState<WindowedSignature[]>([]);
  const [lasCurveMapping, setLasCurveMapping] = useState<Record<string, string>>({});

  // Cycle 10: Master Dashboard state
  const [dashboardData, setDashboardData] = useState<DashboardSummary | null>(null);
  const [dashboardError, setDashboardError] = useState<string | null>(null);

  // Network (PRN) state
  const [networkChannels, setNetworkChannels] = usePersistentState<string[]>(
    'prn-channels', ['WOB', 'RPM', 'ROP', 'TRQ', 'SPP', 'HKLD']
  );
  const [selectedNetworkNode, setSelectedNetworkNode] = useState<string | null>(null);
  const [networkStats, setNetworkStats] = useState<NetworkStats | null>(null);
  const [networkGraphData, setNetworkGraphData] = useState<NetworkGraph | null>(null);

  const handleNetworkStatsUpdate = useCallback((stats: NetworkStats) => {
    setNetworkStats(stats);
  }, []);

  const handleNetworkNodeClick = useCallback((nodeId: string) => {
    setSelectedNetworkNode(prev => prev === nodeId ? null : nodeId);
  }, []);

  // Recent values for detail card sparkline
  const recentValuesForNode = useMemo(() => {
    if (!selectedNetworkNode || data.length === 0) return [];
    const fieldMap: Record<string, keyof DrillingRecord> = {
      WOB: 'wob', TRQ: 'torque', RPM: 'rpm', HKLD: 'hookload',
      SPP: 'spp', ROP: 'rop', DEPTH: 'depth',
    };
    const field = fieldMap[selectedNetworkNode];
    if (!field) return [];
    return data.slice(-50).map(r => Number(r[field]) || 0);
  }, [selectedNetworkNode, data]);

  // Cycle 9: LAS Analyzer callbacks
  const handleAnalyzerWindowData = useCallback(
    (records: Record<string, number>[], regime: LASAnalyzeWindowResponse['regime'], sigs: WindowedSignature[]) => {
      setLasWindowRecords(records);
      setLasWindowRegime(regime);
      setLasWindowSigs(sigs);
    },
    []
  );

  const handleAnalyzerHistoryUpdate = useCallback((history: AnalyzerHistoryEntry[]) => {
    setLasAnalyzerHistory(history);
  }, []);

  // Cycle 10: Build curve mapping from LAS auto-map fields
  const handleLASCurveDataLoadedOuter = useCallback((fileState: LASFileState, selectedCurves: string[]) => {
    const mapping: Record<string, string> = {};
    for (const curve of fileState.curves) {
      if (curve.auto_map_field && selectedCurves.includes(curve.mnemonic)) {
        mapping[curve.auto_map_field] = curve.mnemonic;
      }
    }
    setLasCurveMapping(mapping);
  }, []);

  const handleSurveyLoaded = (newPoints: any[], type: 'plan' | 'actual' | 'offset') => {
    if (type === 'plan') {
      setPlan(newPoints);
    } else if (type === 'actual') {
      setSurveys(prev => [...prev, ...newPoints]);
    }
  };

  const handleLASDataLoaded = (records: DrillingRecord[], mode: 'replace' | 'overlay') => {
    if (mode === 'replace') {
      setData(records);
      if (records.length > 0) setCurrentDepth(records[0].depth);
      setLasOverlayData(null);
    } else {
      setLasOverlayData(records);
    }
  };

  const handleLASCurveDataLoaded = async (fileState: LASFileState, selectedCurves: string[]) => {
    setLasFileState(fileState);
    setLasSelectedCurves(selectedCurves);
    handleLASCurveDataLoadedOuter(fileState, selectedCurves);
    try {
      const res = await fetch(getApiUrl(`/api/v1/las/${fileState.file_id}/curves`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ curves: selectedCurves, max_points: 5000 }),
      });
      if (res.ok) {
        const curveData = await res.json() as LASCurveData;
        setLasCurveData(curveData);
      }
    } catch { /* ignore */ }
  };

  useEffect(() => {
    fetch('/data/drilling_data.json')
      .then(res => res.json())
      .then(data => {
        setData(data);
        if (data.length > 0) setCurrentDepth(data[0].depth);
        setLoading(false);
      })
      .catch(err => console.error("Failed to load data:", err));
  }, []);

  const [offsetWells, setOffsetWells] = useState<OffsetWell[]>([]);
  const [backendResponse, setBackendResponse] = useState<BackendRegimeResponse | null>(null);

  useEffect(() => {
    if (data.length > 0 && offsetWells.length === 0) {
      setOffsetWells(OffsetWellService.generateSimulated(data));
    }
  }, [data]);

  // Single backend call — authoritative source for regime, confidence, color, Betti, recommendation
  useEffect(() => {
    if (data.length < 10) return;
    const records = data.slice(-ANALYSIS_WINDOW).map(r => ({
      wob: r.wob, rpm: r.rpm, rop: r.rop, torque: r.torque, spp: r.spp,
    }));
    fetch(getApiUrl('/api/v1/drilling/ingest'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ records, window_size: ANALYSIS_WINDOW }),
    })
      .then(res => res.json())
      .then((resp: BackendRegimeResponse) => setBackendResponse(resp))
      .catch((e: unknown) => console.warn('Fetch failed:', e));
  }, [data]);

  // Cycle 2: Fetch windowed signatures + change detection when we have enough data
  useEffect(() => {
    if (data.length < 30) return;
    const fields = ['wob', 'rpm', 'rop', 'torque', 'spp'] as const;
    const pointCloud = data.slice(-100).map(r =>
      fields.map(f => Number(r[f]) || 0)
    );

    // Windowed signatures
    fetch(getApiUrl('/api/v1/tda/windowed-signatures'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ point_cloud: pointCloud, window_size: 20, stride: 5 }),
    })
      .then(res => res.ok ? res.json() : null)
      .then(resp => { if (resp?.windows) setWindowedSignatures(resp.windows); })
      .catch((e: unknown) => console.warn('Fetch failed:', e));

    // Change detection between first and second half
    if (pointCloud.length >= 20) {
      const mid = Math.floor(pointCloud.length / 2);
      fetch(getApiUrl('/api/v1/tda/change-detect'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          point_cloud_a: pointCloud.slice(0, mid),
          point_cloud_b: pointCloud.slice(mid),
          threshold: 0.5,
        }),
      })
        .then(res => res.ok ? res.json() : null)
        .then(resp => { if (resp) setChangeResult(resp); })
        .catch((e: unknown) => console.warn('Fetch failed:', e));
    }
  }, [data]);

  // Cycle 3: Fetch curvature field + geodesic when geometry tab is active
  useEffect(() => {
    if (activeTab !== 'geometry' || data.length < 5) return;

    const records = data.slice(-100).map(r => ({
      depth: r.depth, rop: r.rop, wob: r.wob, rpm: r.rpm,
    }));

    // Curvature field from data-driven ROP
    fetch(getApiUrl('/api/v1/geometry/curvature-field'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ records, resolution: 25 }),
    })
      .then(res => res.ok ? res.json() : null)
      .then(resp => {
        if (resp?.points) {
          setCurvaturePoints(resp.points);
          setCurvatureTValues(resp.t_values);
          setCurvatureDValues(resp.d_values);
          setCurvatureResolution(resp.resolution);
        }
      })
      .catch((e: unknown) => console.warn('Fetch failed:', e));

    // Geodesic from first to last data point depth
    const depths = records.map(r => r.depth);
    const minD = Math.min(...depths);
    const maxD = Math.max(...depths);
    if (maxD - minD > 1) {
      fetch(getApiUrl('/api/v1/geometry/geodesic'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          start: [0, minD],
          end: [records.length * 10, maxD],
          n_steps: 100,
        }),
      })
        .then(res => res.ok ? res.json() : null)
        .then(resp => { if (resp?.path) setGeodesicResult(resp); })
        .catch((e: unknown) => console.warn('Fetch failed:', e));
    }
  }, [activeTab, data]);

  // Cycle 4: Fetch fingerprint + attribution when fingerprint tab is active
  useEffect(() => {
    if (activeTab !== 'fingerprint' || data.length < 10) return;
    const fields = ['wob', 'rpm', 'rop', 'torque', 'spp'] as const;
    const pointCloud = data.slice(-100).map(r =>
      fields.map(f => Number(r[f]) || 0)
    );

    // Fingerprint
    fetch(getApiUrl('/api/v1/tda/fingerprint'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ point_cloud: pointCloud }),
    })
      .then(res => res.ok ? res.json() : null)
      .then(resp => {
        if (resp) {
          setFingerprintData(resp);
          // Attribution for the matched regime
          fetch(getApiUrl('/api/v1/tda/attribute'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ point_cloud: pointCloud }),
          })
            .then(res2 => res2.ok ? res2.json() : null)
            .then(attr => { if (attr) setAttributionData(attr); })
            .catch((e: unknown) => console.warn('Fetch failed:', e));

          // Compare matched regime vs second-closest
          if (resp.all_distances) {
            const sorted = Object.entries(resp.all_distances)
              .sort(([, a], [, b]) => (a as number) - (b as number));
            if (sorted.length >= 2) {
              fetch(getApiUrl('/api/v1/tda/compare-regimes'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  point_cloud: pointCloud,
                  regime_a: sorted[0][0],
                  regime_b: sorted[1][0],
                }),
              })
                .then(res3 => res3.ok ? res3.json() : null)
                .then(cmp => { if (cmp) setRegimeCompareData(cmp); })
                .catch((e: unknown) => console.warn('Fetch failed:', e));
            }
          }
        }
      })
      .catch((e: unknown) => console.warn('Fetch failed:', e));
  }, [activeTab, data]);

  // Cycle 5: Fetch topology forecast + transition probabilities when forecast tab is active
  useEffect(() => {
    if (activeTab !== 'forecast' || data.length < 30) return;
    const fields = ['wob', 'rpm', 'rop', 'torque', 'spp'] as const;
    const pointCloud = data.slice(-100).map(r =>
      fields.map(f => Number(r[f]) || 0)
    );

    // Forecast
    fetch(getApiUrl('/api/v1/tda/forecast'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ point_cloud: pointCloud, window_size: 20, stride: 5, n_ahead: 5 }),
    })
      .then(res => res.ok ? res.json() : null)
      .then(resp => { if (resp) setForecastData(resp); })
      .catch((e: unknown) => console.warn('Fetch failed:', e));

    // Transition probabilities
    fetch(getApiUrl('/api/v1/tda/transition-probability'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ point_cloud: pointCloud, window_size: 20, stride: 5 }),
    })
      .then(res => res.ok ? res.json() : null)
      .then(resp => { if (resp) setTransitionData(resp); })
      .catch((e: unknown) => console.warn('Fetch failed:', e));
  }, [activeTab, data]);

  // Cycle 6: Fetch shadow embedding + attractor analysis when shadow tab is active
  useEffect(() => {
    if (activeTab !== 'shadow' || data.length < 10) return;
    const records = data.slice(-200).map(r => ({
      wob: r.wob, rpm: r.rpm, rop: r.rop, torque: r.torque, spp: r.spp,
      hookload: r.hookload, depth: r.depth,
    }));

    // Delay embedding
    fetch(getApiUrl('/api/v1/shadow/embed'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ records, parameter: 'rop', embedding_dim: 3, delay_lag: 1 }),
    })
      .then(res => res.ok ? res.json() : null)
      .then(resp => { if (resp) setShadowEmbedding(resp); })
      .catch((e: unknown) => console.warn('Fetch failed:', e));

    // Attractor analysis
    fetch(getApiUrl('/api/v1/shadow/attractor'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ records, parameter: 'rop', embedding_dim: 3, delay_lag: 1, recurrence_threshold: 0.1 }),
    })
      .then(res => res.ok ? res.json() : null)
      .then(resp => { if (resp) setAttractorAnalysis(resp); })
      .catch((e: unknown) => console.warn('Fetch failed:', e));
  }, [activeTab, data]);

  // Cycle 7: Fetch advisory recommendation + risk when advisory tab is active
  useEffect(() => {
    if (activeTab !== 'advisory' || data.length < 10 || !backendResponse) return;

    const last = data[data.length - 1];
    const currentParams = {
      wob: last.wob, rpm: last.rpm, rop: last.rop,
      torque: last.torque, spp: last.spp,
    };

    // Pick target: OPTIMAL if not already, else NORMAL
    const currentRegime = backendResponse.regime;
    const targetRegime = currentRegime === 'OPTIMAL' ? 'NORMAL' : 'OPTIMAL';

    // Advisory recommendation
    fetch(getApiUrl('/api/v1/advisory/recommend'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        current_params: currentParams,
        current_regime: currentRegime,
        target_regime: targetRegime,
      }),
    })
      .then(res => res.ok ? res.json() : null)
      .then(resp => {
        if (resp) {
          setAdvisoryData(resp);

          // Compute risk for the first step's proposed changes
          const proposed: Record<string, number> = {};
          for (const step of resp.steps) {
            proposed[step.parameter] = step.change_amount;
          }

          fetch(getApiUrl('/api/v1/advisory/risk'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              current_params: currentParams,
              proposed_changes: proposed,
              current_regime: currentRegime,
            }),
          })
            .then(res2 => res2.ok ? res2.json() : null)
            .then(riskResp => { if (riskResp) setRiskData(riskResp); })
            .catch((e: unknown) => console.warn('Fetch failed:', e));
        }
      })
      .catch((e: unknown) => console.warn('Fetch failed:', e));
  }, [activeTab, data, backendResponse]);

  // Cycle 8: Register wells + fetch atlas when field tab is active
  useEffect(() => {
    if (activeTab !== 'field' || data.length < 10) return;

    const registerAndFetch = async () => {
      // Register current well if not yet done
      if (!fieldRegistered) {
        const fields = ['wob', 'rpm', 'rop', 'torque', 'spp', 'depth'] as const;
        const records = data.slice(-200).map(r => {
          const rec: Record<string, number> = {};
          for (const f of fields) rec[f] = Number(r[f]) || 0;
          return rec;
        });

        // Register current well
        try {
          await fetch(getApiUrl('/api/v1/field/register'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: 'Current Well', records }),
          });
        } catch { /* ignore */ }

        // Register simulated offset wells with perturbed data
        const offsetNames = ['Offset A-1', 'Offset B-2', 'Offset C-3'];
        for (let i = 0; i < Math.min(offsetNames.length, 3); i++) {
          const perturbed = records.map(r => {
            const pr: Record<string, number> = {};
            for (const k of Object.keys(r)) {
              pr[k] = (r[k] ?? 0) * (0.85 + Math.random() * 0.3);
            }
            return pr;
          });
          try {
            await fetch(getApiUrl('/api/v1/field/register'), {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                name: offsetNames[i],
                records: perturbed,
                metadata: { type: 'offset', offset_index: i },
              }),
            });
          } catch { /* ignore */ }
        }

        setFieldRegistered(true);
      }

      // Fetch atlas
      try {
        const res = await fetch(getApiUrl('/api/v1/field/atlas'));
        if (res.ok) {
          const resp = await res.json();
          setFieldAtlasData(resp);
        }
      } catch { /* ignore */ }
    };

    registerAndFetch();
  }, [activeTab, data, fieldRegistered]);

  // Cycle 8: Compare selected wells
  useEffect(() => {
    const [idA, idB] = fieldSelectedWells;
    if (!idA || !idB) { setFieldCompareData(null); return; }

    fetch(getApiUrl('/api/v1/field/compare'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ well_id_a: idA, well_id_b: idB }),
    })
      .then(res => res.ok ? res.json() : null)
      .then(resp => { if (resp) setFieldCompareData(resp); })
      .catch((e: unknown) => console.warn('Fetch failed:', e));
  }, [fieldSelectedWells]);

  // Cycle 8: Well selection handler
  const handleFieldWellSelect = useCallback((wellId: string) => {
    setFieldSelectedWells(prev => {
      if (prev[0] === wellId) return [null, prev[1]];
      if (prev[1] === wellId) return [prev[0], null];
      if (!prev[0]) return [wellId, prev[1]];
      if (!prev[1]) return [prev[0], wellId];
      return [wellId, null]; // replace first
    });
  }, []);

  // Cycle 10: Fetch dashboard summary when dashboard tab is active
  useEffect(() => {
    if (activeTab !== 'dashboard' || data.length < 10) return;
    const records = data.slice(-ANALYSIS_WINDOW).map(r => ({
      wob: r.wob, rpm: r.rpm, rop: r.rop, torque: r.torque, spp: r.spp,
    }));
    fetch(getApiUrl('/api/v1/dashboard/summary'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ records, window_size: ANALYSIS_WINDOW }),
    })
      .then(res => {
        if (!res.ok) throw new Error(`${res.status}`);
        return res.json();
      })
      .then(resp => { setDashboardData(resp); setDashboardError(null); })
      .catch(() => { setDashboardError('Backend unavailable \u2014 start the API server'); });
  }, [activeTab, data]);

  // Local vibration analysis (CV-based, separate from TDA regime classification)
  const optimizationResult = useMemo<OptimizationResult | null>(() => {
    if (data.length < 10) return null;
    const window = data.slice(-ANALYSIS_WINDOW);
    return GTMoeOptimizer.optimize(bhaConfig, window);
  }, [data, bhaConfig]);

  // Derived views from backend response
  const regimeStatus = backendResponse
    ? { regime: backendResponse.regime, confidence: backendResponse.confidence, color: backendResponse.color }
    : null;

  const backendBetti = backendResponse
    ? { b0: backendResponse.betti_0, b1: backendResponse.betti_1 }
    : null;

  // Show the latest data point by default
  const displayRecord = data.length > 0 ? data[data.length - 1] : null;

  // Determine CTS pipeline active stage
  const ctsStage: CTSStage = useMemo(() => {
    if (!backendResponse) return 'environment';
    if (backendResponse.persistence_diagram) return 'topology';
    if (backendResponse.recommendation) return 'agency';
    return 'integration';
  }, [backendResponse]);

  // Survey data for DirectionalStudio
  const surveyData = useMemo(() => {
    if (surveys.length > 0) return surveys;
    return data.map(d => ({
      md: d.depth,
      inc: Math.min(d.depth / 100, 90),
      azi: 45 + Math.sin(d.depth / 500) * 10,
      tvd: d.depth * Math.cos(Math.min(d.depth / 100, 90) * Math.PI / 180),
      n_s: d.depth * Math.sin(Math.min(d.depth / 100, 90) * Math.PI / 180) * Math.cos(45 * Math.PI / 180),
      e_w: d.depth * Math.sin(Math.min(d.depth / 100, 90) * Math.PI / 180) * Math.sin(45 * Math.PI / 180),
      dls: 0
    }));
  }, [surveys, data]);

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center bg-background text-primary">
        <div className="flex flex-col items-center gap-4">
          <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin" />
          <p className="font-mono text-lg animate-pulse">INITIALIZING CTS OPERATOR INTERFACE...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col bg-background text-foreground font-sans overflow-hidden selection:bg-primary selection:text-primary-foreground">
      {/* ─── Header ─── */}
      <header className="h-10 border-b border-border bg-card/50 flex items-center px-4 shrink-0 gap-3">
        {/* Left: Title */}
        <div className="flex items-center gap-2 shrink-0">
          <div className="w-2 h-2 bg-primary rounded-full animate-pulse" />
          <span className="text-xs font-mono font-bold tracking-tight">CTS OPERATOR INTERFACE</span>
        </div>

        {/* Regime indicator */}
        {regimeStatus && (
          <div className="flex items-center gap-1.5 ml-2">
            <div className={cn(
              "w-2 h-2 rounded-full",
              regimeStatus.color === 'GREEN' && "bg-green-500",
              regimeStatus.color === 'YELLOW' && "bg-yellow-500",
              regimeStatus.color === 'ORANGE' && "bg-orange-500",
              regimeStatus.color === 'RED' && "bg-red-500",
            )} />
            <span className={cn(
              "text-[10px] font-mono",
              regimeStatus.color === 'GREEN' && "text-green-400",
              regimeStatus.color === 'YELLOW' && "text-yellow-400",
              regimeStatus.color === 'ORANGE' && "text-orange-400",
              regimeStatus.color === 'RED' && "text-red-400",
            )}>
              {regimeStatus.regime} ({(regimeStatus.confidence * 100).toFixed(0)}%)
            </span>
          </div>
        )}

        {/* Depth */}
        <span className="text-[10px] font-mono text-muted-foreground ml-2">
          DEPTH: {currentDepth.toFixed(1)} ft
        </span>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Tab navigation */}
        <div className="flex items-center gap-0.5 mr-3">
          {([
            ['dashboard', 'DASHBOARD'],
            ['cts', 'CTS'],
            ['wellpath', 'WELL PATH'],
            ['wiremesh', 'WIRE MESH'],
            ['network', 'NETWORK'],
            ['geometry', 'SENSITIVITY'],
            ['fingerprint', 'SIGNATURE'],
            ['shadow', 'DYNAMICS'],
            ['forecast', 'FORECAST'],
            ['advisory', 'ADVISORY'],
            ['field', 'FIELD MAP'],
            ['analyzer', 'ANALYZER'],
          ] as [ViewTab, string][]).map(([tab, label]) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={cn(
                'px-2 py-0.5 text-[10px] font-mono transition-colors',
                activeTab === tab
                  ? 'bg-primary/20 text-primary border border-primary/40'
                  : 'text-muted-foreground hover:text-foreground'
              )}
            >
              {label}
            </button>
          ))}
        </div>

        <div className="h-4 w-px bg-border" />

        {/* LAS + Survey buttons */}
        <Dialog>
          <DialogTrigger asChild>
            <Button variant="default" size="sm" className="font-mono text-[10px] h-6 px-2">
              <FileDown className="w-3 h-3 mr-1" />
              LAS
            </Button>
          </DialogTrigger>
          <DialogContent className="sm:max-w-2xl">
            <DialogHeader>
              <DialogTitle className="font-mono">LAS FILE LOADER</DialogTitle>
            </DialogHeader>
            <LASLoader
              onDataLoaded={handleLASDataLoaded}
              onCurveDataLoaded={handleLASCurveDataLoaded}
            />
          </DialogContent>
        </Dialog>
        <Dialog>
          <DialogTrigger asChild>
            <Button variant="ghost" size="sm" className="font-mono text-[10px] h-6 px-2 text-muted-foreground">
              <Upload className="w-3 h-3 mr-1" />
              SURVEYS
            </Button>
          </DialogTrigger>
          <DialogContent className="sm:max-w-lg">
            <DialogHeader>
              <DialogTitle className="font-mono">SURVEY INGESTION</DialogTitle>
            </DialogHeader>
            <SurveyImport onSurveyLoaded={handleSurveyLoaded} />
          </DialogContent>
        </Dialog>
      </header>

      {/* ─── Main Content Area ─── */}
      <div className="flex-1 min-h-0">
        {activeTab === 'dashboard' && (
          <MasterDashboard
            data={dashboardData}
            error={dashboardError}
            onNavigate={(tab) => setActiveTab(tab as ViewTab)}
          />
        )}

        {activeTab === 'analyzer' && (
          <div className="h-full flex flex-col">
            {/* Analyzer controls */}
            {lasFileState && Object.keys(lasCurveMapping).length > 0 ? (
              <>
                <div className="shrink-0 border-b border-border bg-card/30">
                  <LASAnalyzer
                    fileState={lasFileState}
                    curveMapping={lasCurveMapping}
                    onWindowData={handleAnalyzerWindowData}
                    onHistoryUpdate={handleAnalyzerHistoryUpdate}
                  />
                </div>
                <div className="h-10 shrink-0 border-b border-border bg-card/20">
                  <RegimeStrip history={lasAnalyzerHistory} />
                </div>
                <div className="flex-1 min-h-0 flex">
                  {/* Left: Log data from current window (60%) */}
                  <div className="w-[60%] border-r border-border p-2">
                    {lasWindowRecords.length > 0 ? (
                      <div className="h-full flex flex-col gap-2 overflow-y-auto">
                        {/* Window regime header */}
                        {lasWindowRegime && (
                          <div className="flex items-center gap-3 px-2 py-1 bg-card/60 border border-border/50">
                            <div className={cn(
                              'w-3 h-3 rounded-full',
                              lasWindowRegime.color === 'GREEN' && 'bg-green-500',
                              lasWindowRegime.color === 'YELLOW' && 'bg-yellow-500',
                              lasWindowRegime.color === 'ORANGE' && 'bg-orange-500',
                              lasWindowRegime.color === 'RED' && 'bg-red-500',
                            )} />
                            <span className="text-xs font-mono">{lasWindowRegime.regime}</span>
                            <span className="text-[10px] font-mono text-muted-foreground">
                              {(lasWindowRegime.confidence * 100).toFixed(0)}%
                            </span>
                            <span className="text-[10px] font-mono text-muted-foreground ml-auto">
                              {lasWindowRecords.length} records
                            </span>
                          </div>
                        )}
                        {/* Recommendation */}
                        {lasWindowRegime?.recommendation && (
                          <div className="text-[10px] font-mono text-foreground/80 px-2">
                            {lasWindowRegime.recommendation}
                          </div>
                        )}
                        {/* Simple table of last few records */}
                        <div className="flex-1 overflow-auto">
                          <table className="w-full text-[9px] font-mono">
                            <thead>
                              <tr className="text-muted-foreground border-b border-border/30">
                                <th className="text-left px-1 py-0.5">Depth</th>
                                <th className="text-right px-1">WOB</th>
                                <th className="text-right px-1">RPM</th>
                                <th className="text-right px-1">ROP</th>
                                <th className="text-right px-1">Torque</th>
                                <th className="text-right px-1">SPP</th>
                              </tr>
                            </thead>
                            <tbody>
                              {lasWindowRecords.slice(-20).map((r, i) => (
                                <tr key={i} className="border-b border-border/10 hover:bg-card/40">
                                  <td className="text-left px-1 py-0.5">{(r.depth ?? 0).toFixed(1)}</td>
                                  <td className="text-right px-1">{(r.wob ?? 0).toFixed(1)}</td>
                                  <td className="text-right px-1">{(r.rpm ?? 0).toFixed(0)}</td>
                                  <td className="text-right px-1">{(r.rop ?? 0).toFixed(1)}</td>
                                  <td className="text-right px-1">{(r.torque ?? 0).toFixed(0)}</td>
                                  <td className="text-right px-1">{(r.spp ?? 0).toFixed(0)}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    ) : (
                      <div className="h-full flex items-center justify-center text-muted-foreground font-mono text-xs">
                        Use controls above to step through depth
                      </div>
                    )}
                  </div>
                  {/* Right: KPI + Betti for current window (40%) */}
                  <div className="w-[40%] flex flex-col p-2 gap-2 overflow-y-auto">
                    {lasWindowRegime && (
                      <KPICards
                        regime={lasWindowRegime.regime}
                        confidence={lasWindowRegime.confidence}
                        color={lasWindowRegime.color}
                        betti={{ b0: lasWindowRegime.betti_0, b1: lasWindowRegime.betti_1 }}
                        record={lasWindowRecords.length > 0 ? {
                          id: 0, depth: lasWindowRecords[lasWindowRecords.length - 1]?.depth ?? 0,
                          rop: lasWindowRecords[lasWindowRecords.length - 1]?.rop ?? 0,
                          wob: lasWindowRecords[lasWindowRecords.length - 1]?.wob ?? 0,
                          rpm: lasWindowRecords[lasWindowRecords.length - 1]?.rpm ?? 0,
                          hookload: lasWindowRecords[lasWindowRecords.length - 1]?.hookload ?? 0,
                          spp: lasWindowRecords[lasWindowRecords.length - 1]?.spp ?? 0,
                          torque: lasWindowRecords[lasWindowRecords.length - 1]?.torque ?? 0,
                          surprise: 0, activity: 0, activity_type: 'silent' as const,
                          confidence: lasWindowRegime.confidence, mu: [0, 0, 0], sigma_trace: 0,
                        } : null}
                        currentDepth={lasWindowRecords.length > 0 ? (lasWindowRecords[lasWindowRecords.length - 1]?.depth ?? 0) : 0}
                      />
                    )}
                    <TrustGauge
                      confidence={lasWindowRegime?.confidence ?? 0}
                      regime={lasWindowRegime?.regime ?? 'UNKNOWN'}
                    />
                    {lasWindowSigs.length > 0 && (
                      <BettiTimeline windows={lasWindowSigs} />
                    )}
                  </div>
                </div>
              </>
            ) : (
              <div className="h-full flex items-center justify-center text-muted-foreground font-mono text-sm">
                Load a LAS file and map curves to drilling parameters to use the analyzer
              </div>
            )}
          </div>
        )}

        {activeTab === 'cts' && (
          <div className="h-full flex">
            {/* Left Panel: Attractor Manifold (~40%) */}
            <div className="w-[40%] border-r border-border">
              <AttractorManifold
                data={data}
                currentDepth={currentDepth}
                regimeStatus={regimeStatus}
              />
            </div>

            {/* Center Panel: Persistence Barcode + Depth Track (~35%) */}
            <div className="w-[35%] border-r border-border flex flex-col">
              {/* Top: Persistence Barcode */}
              <div className="h-[45%] border-b border-border/50 p-1">
                <PersistenceBarcode
                  diagram={backendResponse?.persistence_diagram ?? null}
                  filtrationRange={
                    backendResponse?.persistence_diagram?.filtration_range ?? [0, 1]
                  }
                />
              </div>
              {/* Bottom: Depth Track or Topological Time Machine */}
              <div className="flex-1 min-h-0 p-1">
                {lasCurveData ? (
                  <div className="h-full overflow-hidden [&_.border-border]:border-0 [&>div]:!border-0 [&>div]:!bg-transparent">
                    <LogTrackViewer data={lasCurveData} currentDepth={currentDepth} height={400} />
                  </div>
                ) : (
                  <div className="h-full flex flex-col">
                    <div className="h-1/2 border-b border-border/30">
                      <BettiTimeline windows={windowedSignatures} />
                    </div>
                    <div className="h-1/2">
                      <TopologicalHeatmap windows={windowedSignatures} />
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Right Panel: Trust + KPIs + Recommendations (~25%) */}
            <div className="w-[25%] flex flex-col p-2 gap-2 overflow-y-auto">
              {/* Trust Gauge */}
              <TrustGauge
                confidence={backendResponse?.confidence ?? 0}
                regime={backendResponse?.regime ?? 'UNKNOWN'}
              />

              {/* KPI Cards */}
              <KPICards
                regime={backendResponse?.regime ?? null}
                confidence={backendResponse?.confidence ?? null}
                color={backendResponse?.color ?? null}
                betti={backendBetti}
                record={displayRecord}
                currentDepth={currentDepth}
              />

              {/* Recommendation */}
              {backendResponse?.recommendation && (
                <div className="bg-card/60 border border-border/50 px-2 py-1.5">
                  <div className="text-[9px] font-mono text-muted-foreground uppercase tracking-wider mb-0.5">
                    Recommendation
                  </div>
                  <div className="text-[11px] font-mono leading-tight text-foreground/90">
                    {backendResponse.recommendation}
                  </div>
                </div>
              )}

              {/* GT-MoE Summary */}
              {optimizationResult && (
                <div className="bg-card/60 border border-border/50 px-2 py-1.5">
                  <div className="text-[9px] font-mono text-muted-foreground uppercase tracking-wider mb-0.5">
                    GT-MoE Optimizer
                  </div>
                  <div className="text-[11px] font-mono leading-tight space-y-0.5">
                    <div>
                      <span className="text-muted-foreground">Vibration: </span>
                      <span className={cn(
                        optimizationResult.regime === 'Normal' ? 'text-green-400' :
                        optimizationResult.regime === 'Stick-Slip' ? 'text-orange-400' :
                        optimizationResult.regime === 'Whirl' ? 'text-yellow-400' :
                        'text-red-400'
                      )}>
                        {optimizationResult.regime}
                      </span>
                    </div>
                    <div className="text-foreground/80 text-[10px]">
                      {optimizationResult.reasoning}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'wellpath' && (
          <div className="h-full flex">
            <div className="w-[60%] h-full border-r border-border">
              <DirectionalStudio
                surveys={surveyData}
                plan={plan}
                offsetWells={offsetWells}
              />
            </div>
            <div className="w-[40%] h-full flex flex-col">
              <div className="flex-1 min-h-0">
                <BHABuilder config={bhaConfig} onChange={setBhaConfig} />
              </div>
              <div className="h-[40%] border-t border-border">
                <GTMoePanel result={optimizationResult} backendRegime={backendResponse?.regime} />
              </div>
            </div>
          </div>
        )}

        {activeTab === 'wiremesh' && (
          <div className="h-full">
            {lasCurveData ? (
              <LASMeshViz data={lasCurveData} selectedCurves={lasSelectedCurves} />
            ) : (
              <div className="h-full flex items-center justify-center text-muted-foreground font-mono text-sm">
                Load a LAS file to view wire mesh visualization
              </div>
            )}
          </div>
        )}

        {activeTab === 'network' && (
          <div className="h-full flex flex-col">
            {/* Toolbar */}
            <div className="h-10 border-b border-border bg-card/30 flex items-center px-3 gap-3 shrink-0">
              <ChannelSelector
                selected={networkChannels}
                onChange={setNetworkChannels}
              />
              <span className="text-[10px] font-mono text-muted-foreground">
                {networkChannels.length} channels selected
              </span>
            </div>

            {/* Graph area */}
            <div className="flex-1 min-h-0 relative">
              <ParameterNetworkGraph
                data={data}
                lasCurveData={lasCurveData}
                selectedChannels={networkChannels}
                onNodeClick={handleNetworkNodeClick}
                onStatsUpdate={handleNetworkStatsUpdate}
                onGraphData={setNetworkGraphData}
              />

              {/* Detail card overlay */}
              {selectedNetworkNode && networkStats && (
                (() => {
                  // Find the node data from the last fetch
                  const node = networkGraphData?.nodes.find(n => n.id === selectedNetworkNode);
                  if (!node) return null;
                  return (
                    <ParameterDetailCard
                      node={node}
                      edges={networkGraphData?.edges ?? []}
                      allNodes={networkGraphData?.nodes ?? []}
                      recentValues={recentValuesForNode}
                      onClose={() => setSelectedNetworkNode(null)}
                      onChipClick={handleNetworkNodeClick}
                      className="top-4 right-4"
                    />
                  );
                })()
              )}
            </div>

            {/* Stats bar */}
            <NetworkStatsBar stats={networkStats} className="h-8 shrink-0" />
          </div>
        )}

        {activeTab === 'geometry' && (
          <div className="h-full flex">
            {/* Left: Curvature Field (60%) */}
            <div className="w-[60%] border-r border-border">
              <CurvatureField
                points={curvaturePoints}
                tValues={curvatureTValues}
                dValues={curvatureDValues}
                resolution={curvatureResolution}
              />
            </div>
            {/* Right: Geodesic Overlay (40%) */}
            <div className="w-[40%]">
              <GeodesicOverlay
                geodesic={geodesicResult}
                tRange={curvatureTValues.length >= 2
                  ? [curvatureTValues[0], curvatureTValues[curvatureTValues.length - 1]]
                  : [0, 100]}
                dRange={curvatureDValues.length >= 2
                  ? [curvatureDValues[0], curvatureDValues[curvatureDValues.length - 1]]
                  : [0, 5000]}
              />
            </div>
          </div>
        )}

        {activeTab === 'fingerprint' && (
          <div className="h-full flex">
            {/* Left: Regime Fingerprint radar (35%) */}
            <div className="w-[35%] border-r border-border">
              <RegimeFingerprint fingerprint={fingerprintData} />
            </div>
            {/* Center: Attribution Bars (30%) */}
            <div className="w-[30%] border-r border-border">
              <AttributionBars attribution={attributionData} />
            </div>
            {/* Right: Regime Compare (35%) */}
            <div className="w-[35%]">
              <RegimeCompare comparison={regimeCompareData} />
            </div>
          </div>
        )}

        {activeTab === 'shadow' && (
          <div className="h-full flex">
            {/* Left: Delay Embedding 3D scatter (60%) */}
            <div className="w-[60%] border-r border-border">
              <DelayEmbedding embedding={shadowEmbedding} />
            </div>
            {/* Right: Lyapunov Indicator + Attractor Analysis (40%) */}
            <div className="w-[40%]">
              <LyapunovIndicator analysis={attractorAnalysis} />
            </div>
          </div>
        )}

        {activeTab === 'forecast' && (
          <div className="h-full flex">
            {/* Left: Topology Forecast trajectory (60%) */}
            <div className="w-[60%] border-r border-border">
              <TopologyForecast forecast={forecastData} history={windowedSignatures} />
            </div>
            {/* Right: Transition Radar (40%) */}
            <div className="w-[40%]">
              <TransitionRadar transition={transitionData} />
            </div>
          </div>
        )}

        {activeTab === 'advisory' && (
          <div className="h-full flex">
            {/* Left: Geodesic Navigator path visualization (55%) */}
            <div className="w-[55%] border-r border-border">
              <GeodesicNavigator advisory={advisoryData} />
            </div>
            {/* Right: Advisory Panel step-by-step prescription (45%) */}
            <div className="w-[45%]">
              <AdvisoryPanel advisory={advisoryData} risk={riskData} />
            </div>
          </div>
        )}

        {activeTab === 'field' && (
          <div className="h-full flex">
            {/* Left: Field Atlas well grid (45%) */}
            <div className="w-[45%] border-r border-border">
              <FieldAtlas
                atlas={fieldAtlasData}
                onSelectWell={handleFieldWellSelect}
                selectedWellIds={fieldSelectedWells}
              />
            </div>
            {/* Right: Well Compare (55%) */}
            <div className="w-[55%]">
              <WellCompare
                comparison={fieldCompareData}
                wellA={fieldAtlasData?.wells.find(w => w.well_id === fieldSelectedWells[0]) ?? null}
                wellB={fieldAtlasData?.wells.find(w => w.well_id === fieldSelectedWells[1]) ?? null}
              />
            </div>
          </div>
        )}
      </div>

      {/* ─── Footer: CTS Pipeline + Change Detector ─── */}
      <div className="h-8 shrink-0 flex border-t border-border">
        <CTSPipelineBar activeStage={ctsStage} className="flex-1" />
        <div className="w-px bg-border" />
        <ChangeDetector changeResult={changeResult} className="shrink-0" />
      </div>

      {/* DrillDown Modal (kept for depth track click) */}
      <DrillDownModal
        record={selectedRecord}
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
      />
    </div>
  );
}
