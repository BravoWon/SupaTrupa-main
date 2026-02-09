export interface DrillingRecord {
  id: number;
  timestamp?: number;
  depth: number;
  rop: number;
  wob: number;
  rpm: number;
  hookload: number;
  spp: number;
  torque: number;
  surprise: number;
  activity: number;
  activity_type: 'spike' | 'subthreshold' | 'silent';
  confidence: number;
  mu: number[];
  sigma_trace: number;
}

export interface DrillingDataset {
  records: DrillingRecord[];
  stats: {
    maxDepth: number;
    minDepth: number;
    maxSurprise: number;
    avgConfidence: number;
    totalSamples: number;
  };
}

import { BHAConfig } from './lib/gtMoeOptimizer';
import { OffsetWell } from './lib/offsetWells';
import { DownlinkCommand } from './lib/rssDownlink';
import type { CurveCategory } from './lib/lasMnemonics';

// --- LAS File Types ---

export interface LASCurve {
  mnemonic: string;
  unit: string;
  description: string;
  category: CurveCategory;
  is_numeric: boolean;
  null_pct: number;
  min_val: number | null;
  max_val: number | null;
  auto_map_field: string | null;
}

export interface LASFileState {
  file_id: string;
  well_name: string;
  company: string;
  index_type: 'DEPTH' | 'TIME';
  index_unit: string;
  index_min: number;
  index_max: number;
  num_rows: number;
  num_curves: number;
  curves: LASCurve[];
}

export interface LASCurveData {
  index_name: string;
  index_unit: string;
  index_values: number[];
  curves: Record<string, { values: (number | null)[]; unit: string; category: string }>;
  total_rows: number;
  decimation_factor: number;
}

export interface LASTrackConfig {
  id: string;
  label: string;
  curves: string[];
  color: string;
  scaleMin?: number;
  scaleMax?: number;
  logScale?: boolean;
}

// =============================================================================
// Parameter Resonance Network (PRN) Types
// =============================================================================

export type ParameterCategory =
  | 'mechanical'
  | 'hydraulic'
  | 'formation'
  | 'directional'
  | 'vibration'
  | 'performance';

export type HealthStatus = 'optimal' | 'caution' | 'warning' | 'critical';

export interface ParameterNode {
  id: string;
  full_name: string;
  category: ParameterCategory;
  current_value: number;
  unit: string;
  health: HealthStatus;
  anomaly_flag: boolean;
  z_score: number;
  dominant_frequency_hz: number;
  mean: number;
  std: number;
  importance: number;
}

export interface CorrelationEdge {
  source: string;
  target: string;
  pearson_r: number;
  is_significant: boolean;
}

export interface NetworkGraph {
  nodes: ParameterNode[];
  edges: CorrelationEdge[];
  strong_count: number;
  anomaly_count: number;
  system_health: string;
  computation_time_ms: number;
}

export interface NetworkStats {
  nodeCount: number;
  edgeCount: number;
  strongCount: number;
  anomalyCount: number;
  systemHealth: string;
  computationTimeMs: number;
}

// =============================================================================
// Cycle 2: Topological Time Machine Types
// =============================================================================

export interface BettiCurveData {
  h0: { curve: number[]; t_values: number[] };
  h1: { curve: number[]; t_values: number[] };
}

export interface WindowedSignature {
  window_index: number;
  betti_0: number;
  betti_1: number;
  entropy_h0: number;
  entropy_h1: number;
  total_persistence_h0: number;
  total_persistence_h1: number;
}

export interface WindowedSignatureResponse {
  windows: WindowedSignature[];
  window_size: number;
  stride: number;
  num_windows: number;
}

export interface ChangeDetectResult {
  detected_change: boolean;
  change_magnitude: number;
  betti_change: Record<string, number>;
  landscape_distance: Record<string, number>;
  silhouette_distance: Record<string, number>;
}

// =============================================================================
// Cycle 4: Persistence Fingerprinting Types
// =============================================================================

export interface FingerprintDriver {
  feature: string;
  contribution_pct: number;
  observed: number;
  signature: number;
  direction: 'higher' | 'lower';
}

export interface FingerprintResponse {
  matched_regime: string;
  confidence: number;
  is_transition: boolean;
  observed_features: Record<string, number>;
  matched_signature: Record<string, number>;
  feature_distances: Record<string, number>;
  top_drivers: FingerprintDriver[];
  all_distances: Record<string, number>;
}

export interface Attribution {
  feature: string;
  observed: number;
  signature: number;
  normalized_distance: number;
  squared_contribution: number;
  contribution_pct: number;
}

export interface AttributionResponse {
  regime: string;
  confidence: number;
  attributions: Attribution[];
  total_distance: number;
  dominant_dimension: string;
  interpretation: string;
}

export interface RegimeCompareResponse {
  regime_a: string;
  regime_b: string;
  features_a: Record<string, number>;
  features_b: Record<string, number>;
  feature_deltas: Record<string, number>;
  topological_distance: number;
  discriminating_features: string[];
  interpretation: string;
}

// =============================================================================
// Cycle 3: Manifold Geometry Engine Types
// =============================================================================

export interface MetricFieldPoint {
  t: number;
  d: number;
  g_tt: number;
  g_dd: number;
  determinant: number;
  ricci_scalar: number;
  rop: number;
}

export interface MetricFieldResponse {
  points: MetricFieldPoint[];
  t_values: number[];
  d_values: number[];
  resolution: number;
}

export interface GeodesicResponse {
  path: [number, number][];
  total_length: number;
  start_rop: number;
  end_rop: number;
  start_curvature: number;
  end_curvature: number;
}

export interface CurvatureFieldResponse {
  points: MetricFieldPoint[];
  t_values: number[];
  d_values: number[];
  resolution: number;
  max_curvature: number;
  min_curvature: number;
  mean_curvature: number;
}

// =============================================================================
// Cycle 5: Predictive Topology Types
// =============================================================================

export interface ForecastPoint {
  window_index: number;
  betti_0: number;
  betti_1: number;
  entropy_h0: number;
  entropy_h1: number;
  total_persistence_h0: number;
  total_persistence_h1: number;
  confidence_upper: Record<string, number>;
  confidence_lower: Record<string, number>;
}

export interface ForecastResponse {
  current: Record<string, number>;
  forecast: ForecastPoint[];
  velocity: Record<string, number>;
  acceleration: Record<string, number>;
  trend_direction: string;
  stability_index: number;
  n_windows_used: number;
  n_ahead: number;
}

export interface TransitionProbResponse {
  current_regime: string;
  probabilities: Record<string, number>;
  trending_toward: string;
  trending_away: string;
  velocity_magnitude: number;
  estimated_windows_to_transition: number | null;
  risk_level: string;
}

// =============================================================================
// Cycle 6: Shadow Tensor Integration Types
// =============================================================================

export interface ShadowEmbedResponse {
  point_cloud: number[][];
  metric_proxy: number[];
  tangent_proxy: number[];
  fractal_proxy: number[];
  embedding_dim: number;
  delay_lag: number;
  n_points: number;
  total_dimension: number;
}

export interface AttractorAnalysis {
  lyapunov_exponent: number;
  lyapunov_interpretation: string;
  correlation_dimension: number;
  recurrence_rate: number;
  determinism: number;
  attractor_type: string;
  embedding_dim: number;
  n_points: number;
  laminarity: number;
  trapping_time: number;
}

// =============================================================================
// Cycle 7: Autonomous Advisory Types
// =============================================================================

export interface ParameterStep {
  step_index: number;
  parameter: string;
  current_value: number;
  target_value: number;
  change_amount: number;
  change_pct: number;
  priority: number;
  rationale: string;
}

export interface AdvisoryResponse {
  current_regime: string;
  target_regime: string;
  confidence: number;
  steps: ParameterStep[];
  geodesic_length: number;
  euclidean_length: number;
  path_efficiency: number;
  risk_score: number;
  risk_level: string;
  estimated_transitions: number;
  reasoning: string[];
  parameter_trajectory: Record<string, number>[];
}

export interface RiskFactor {
  factor: string;
  value: string;
  risk: number;
}

export interface RiskAssessmentResponse {
  overall_risk: number;
  risk_level: string;
  risk_factors: RiskFactor[];
  mitigations: string[];
  abort_conditions: string[];
  regime_risk: number;
  path_risk: number;
  correlation_risk: number;
}

// =============================================================================
// Cycle 8: Field-Level Intelligence Types
// =============================================================================

export interface FieldWellEntry {
  well_id: string;
  name: string;
  depth_min: number;
  depth_max: number;
  num_records: number;
  feature_vector: number[];
  regime: string;
  confidence: number;
  regime_distribution: Record<string, number>;
  windowed_betti: { window_index: number; betti_0: number; betti_1: number; regime: string }[];
}

export interface FieldAtlasResponse {
  wells: FieldWellEntry[];
  well_count: number;
  field_summary: {
    well_count: number;
    regime_distribution: Record<string, number>;
    mean_signature: number[];
    signature_spread: number[];
    depth_range: [number, number];
  };
}

export interface FieldCompareResponse {
  well_a: string;
  well_b: string;
  topological_distance: number;
  feature_deltas: Record<string, number>;
  discriminating_features: string[];
  regime_similarity: number;
  depth_overlap: number;
  interpretation: string;
}

export interface PatternMatch {
  well_id: string;
  name: string;
  distance: number;
  regime: string;
  confidence: number;
  feature_vector: number[];
}

// =============================================================================
// Cycle 9: LAS Sliding Analyzer Types
// =============================================================================

export interface LASAnalyzeWindowResponse {
  records: Record<string, number>[];
  count: number;
  regime: {
    regime: string;
    confidence: number;
    color: string;
    betti_0: number;
    betti_1: number;
    recommendation: string;
  } | null;
  windowed_signatures: WindowedSignature[];
  index_range: [number, number];
}

export interface AnalyzerHistoryEntry {
  start: number;
  end: number;
  regime: string;
  confidence: number;
  color: string;
}

// =============================================================================
// Cycle 10: Master Dashboard Types
// =============================================================================

export interface DashboardSummary {
  regime: string;
  regime_display: string;
  confidence: number;
  color: string;
  recommendation: string;
  rop: number;
  wob: number;
  rpm: number;
  torque: number;
  spp: number;
  drilling_zones: number;
  coupling_loops: number;
  signature_stability: number;
  predictability_index: number;
  behavioral_consistency: number;
  trending_toward: string;
  transition_risk: string;
  estimated_windows_to_transition: number | null;
  top_advisory: string | null;
  advisory_risk: string | null;
}

export type MissionState = MissionData;

export interface MissionData {
  version: string;
  timestamp: number;
  name: string;
  surveys: {
    plan: any[];
    actual: any[];
  };
  offsetWells: OffsetWell[];
  bhaConfig: BHAConfig;
  rssHistory: {
    timestamp: number;
    command: DownlinkCommand;
  }[];
  drillingData: DrillingRecord[];
}
