/**
 * Unified Activity:State Platform - Shared TypeScript Types
 * 
 * These types mirror the Python dataclasses in jones_framework,
 * enabling type-safe communication between frontend and backend.
 */

// =============================================================================
// Core Types
// =============================================================================

/**
 * Regime identifiers matching Python RegimeID enum
 */
export type RegimeID =
  | 'DARCY_FLOW'
  | 'NON_DARCY_FLOW'
  | 'TURBULENT'
  | 'MULTIPHASE'
  | 'BIT_BOUNCE'
  | 'PACKOFF'
  | 'OPTIMAL'
  | 'STICK_SLIP'
  | 'WHIRL'
  | 'FORMATION_CHANGE'
  | 'WASHOUT'
  | 'LOST_CIRCULATION'
  | 'NORMAL'
  | 'TRANSITION'
  | 'KICK'
  | 'UNKNOWN';

/**
 * Drilling regime identifiers for GT-MoE
 */
export type DrillingRegime = 'Normal' | 'Stick-Slip' | 'Whirl' | 'Bit Bounce';

/**
 * ConditionState - Atomic immutable data unit
 * Mirrors Python jones_framework.core.condition_state.ConditionState
 */
export interface ConditionState {
  state_id: string;
  timestamp: number;
  vector: number[];
  metadata: Record<string, unknown>;
  verified: boolean;
  dimension: number;
}

/**
 * ActivityState - Macroscopic regime definition
 * Mirrors Python jones_framework.core.activity_state.ActivityState
 */
export interface ActivityState {
  regime_id: RegimeID;
  transition_threshold: number;
  metadata: Record<string, unknown>;
}

/**
 * Persistence diagram representing topological features
 */
export interface PersistenceDiagram {
  h0: [number, number][]; // Birth-death pairs for H0
  h1: [number, number][]; // Birth-death pairs for H1
  h2?: [number, number][]; // Optional H2 features
}

/**
 * TDA features extracted from point cloud
 */
export interface TDAFeatures {
  betti_0: number;
  betti_1: number;
  entropy_h0: number;
  entropy_h1: number;
  max_lifetime_h0: number;
  max_lifetime_h1: number;
  mean_lifetime_h0: number;
  mean_lifetime_h1: number;
  n_features_h0: number;
  n_features_h1: number;
}

// =============================================================================
// Extended TDA Types - Interdimensional Representations
// =============================================================================

/**
 * Persistence Landscape - Functional representation of persistence diagrams
 *
 * The k-th landscape function λ_k(t) is the k-th largest value of tent functions.
 * Enables statistical operations, ML vectorization, stable distances.
 */
export interface PersistenceLandscape {
  landscapes: number[][];  // Shape: [k, resolution] - k landscape functions
  t_values: number[];      // Evaluation points along filtration
  homology_dim: number;    // Which H_n this represents
  birth_death_pairs?: [number, number][];  // Original diagram
}

/**
 * Persistence Silhouette - Weighted power mean of landscape functions
 *
 * φ_p(t) = (Σ_k w_k λ_k(t)^p)^(1/p) / Σ_k w_k
 * Provides single-curve summary of entire persistence structure.
 */
export interface PersistenceSilhouette {
  silhouette: number[];    // Shape: [resolution]
  t_values: number[];      // Evaluation points
  power: number;           // Weighting power (1=arithmetic, 2=RMS)
  homology_dim: number;
  weights?: number[];      // Feature weights used
}

/**
 * Persistence Image - 2D discretization for ML pipelines
 *
 * Converts birth-death pairs to 2D image via:
 * 1. Transform to birth-persistence coords
 * 2. Weight by persistence
 * 3. Convolve with Gaussian kernel
 */
export interface PersistenceImage {
  image: number[][];       // Shape: [resolution, resolution]
  birth_range: [number, number];
  persistence_range: [number, number];
  sigma: number;           // Gaussian kernel bandwidth
  homology_dim: number;
  resolution: number;
}

/**
 * Betti Curve - Betti numbers as function of filtration
 *
 * β_n(t) = number of n-dimensional holes at filtration value t
 */
export interface BettiCurve {
  curve: number[];         // Betti numbers at each t
  t_values: number[];      // Filtration values
  homology_dim: number;
}

/**
 * Complete Topological Signature - Full interdimensional fingerprint
 *
 * Combines all TDA representations at all dimensions for
 * comprehensive topological analysis.
 */
export interface TopologicalSignature {
  // Core persistence data
  diagrams: Record<number, [number, number][]>;  // H_0, H_1, H_2, ...

  // Functional representations
  landscapes: Record<number, PersistenceLandscape>;
  silhouettes: Record<number, PersistenceSilhouette>;
  betti_curves: Record<number, BettiCurve>;

  // Image representation
  images: Record<number, PersistenceImage>;

  // Summary statistics
  betti_numbers: Record<number, number>;
  persistence_entropy: Record<number, number>;
  total_persistence: Record<number, number>;

  // Metadata
  max_dimension: number;
  num_points: number;
}

/**
 * Topological Change Detection Result
 */
export interface TopologicalChangeResult {
  betti_change: Record<number, number>;
  landscape_distance: Record<number, number>;
  silhouette_distance: Record<number, number>;
  detected_change: boolean;
  change_magnitude: number;
}

/**
 * Streaming TDA State - For real-time topological analysis
 *
 * Maintains sliding window of points and computes
 * topological features incrementally.
 */
export interface StreamingTDAState {
  window_size: number;
  max_dim: number;
  current_betti: Record<number, number>;
  update_count: number;
}

/**
 * Classification result from regime classifier
 */
export interface ClassificationResult {
  regime_id: RegimeID;
  confidence: number;
  betti_0: number;
  betti_1: number;
  entropy_h1: number;
  features: TDAFeatures;
}

/**
 * Expert configuration in Mixture of Experts
 */
export interface ExpertConfig {
  regime_id: RegimeID;
  description: string;
  is_active: boolean;
  accuracy?: number;
  sharpe_ratio?: number;
}

/**
 * MoE processing result
 */
export interface MoEProcessResult {
  output: number[];
  active_regime: RegimeID;
  expert_used: string;
}

// =============================================================================
// API Request/Response Types
// =============================================================================

export interface CreateStateRequest {
  vector: number[];
  metadata?: Record<string, unknown>;
  domain?: string;
}

export interface CreateMarketStateRequest {
  price: number;
  volume: number;
  bid: number;
  ask: number;
  symbol: string;
}

export interface ClassifyRequest {
  point_cloud: number[][];
}

export interface ProcessRequest {
  state: CreateStateRequest;
  point_cloud?: number[][];
  auto_swap?: boolean;
}

export interface HotSwapRequest {
  regime_name: string;
}

export interface SystemStatus {
  status: 'online' | 'offline' | 'error';
  framework_available: boolean;
  active_regime: RegimeID | null;
  num_experts: number;
  uptime_seconds: number;
}

// =============================================================================
// WebSocket Message Types
// =============================================================================

export type WebSocketMessageType = 
  | 'classify'
  | 'process'
  | 'ping'
  | 'pong'
  | 'classification'
  | 'processed'
  | 'regime_change';

export interface WebSocketMessage {
  type: WebSocketMessageType;
  [key: string]: unknown;
}

export interface ClassifyMessage extends WebSocketMessage {
  type: 'classify';
  point_cloud: number[][];
}

export interface ProcessMessage extends WebSocketMessage {
  type: 'process';
  vector: number[];
}

export interface ClassificationResponse extends WebSocketMessage {
  type: 'classification';
  regime_id: RegimeID;
  confidence: number;
  betti_0: number;
  betti_1: number;
  timestamp: string;
}

export interface RegimeChangeMessage extends WebSocketMessage {
  type: 'regime_change';
  regime: RegimeID;
  confidence: number;
  timestamp: string;
}

// =============================================================================
// Drilling-Specific Types (from dashboard)
// =============================================================================

export interface BHAConfig {
  bitType: 'PDC' | 'ROLLER_CONE' | 'DIAMOND';
  motorBend: number;
  stabilizers: number;
  stabilizerPos?: number;
  flowRestrictor?: number;
}

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

export interface OptimizationResult {
  regime: DrillingRegime;
  bettiNumbers: { b0: number; b1: number };
  recommendation: BHAConfig;
  confidence: number;
  reasoning: string;
}

export interface OffsetWell {
  id: string;
  name: string;
  points: {
    md: number;
    tvd: number;
    n_s: number;
    e_w: number;
  }[];
}

export interface MissionState {
  version: string;
  timestamp: number;
  name: string;
  surveys: {
    plan: unknown[];
    actual: unknown[];
  };
  offsetWells: OffsetWell[];
  bhaConfig: BHAConfig;
  rssHistory: {
    timestamp: number;
    command: unknown;
  }[];
  drillingData: DrillingRecord[];
}

// =============================================================================
// Temporal-Spatial Coordinate Bridge (Drilling Manifold Geometry)
// =============================================================================

/**
 * Coordinate system for drilling data
 * - TIME: t-indexed (surface data, dynamics, vibrations)
 * - DEPTH: d-indexed (logs, surveys, geology)
 * - DUAL: Both coordinates available (unified manifold)
 */
export type CoordinateSystem = 'TIME' | 'DEPTH' | 'DUAL';

/**
 * ROP Jacobian - the metric tensor component bridging time and depth
 *
 * The relationship: t = ∫(1/ROP) d(depth)  [back integral]
 *                   d = ∫(ROP) d(time)     [forward integral]
 */
export interface ROPJacobian {
  rop_values: number[];
  index_values: number[];
  index_type: CoordinateSystem;
  mean_rop: number;
  rop_variance: number;
}

/**
 * Drilling condition state with dual coordinate support
 *
 * Every drilling observation exists at BOTH a time and a depth,
 * related by the ROP Jacobian.
 */
export interface DrillingConditionState {
  time: number | null;
  depth: number | null;
  rop: number | null;
  features: number[];
  coordinate_system: CoordinateSystem;
  metadata: Record<string, unknown>;
}

/**
 * Coordinate atlas for transforming between time and depth charts
 */
export interface CoordinateAtlas {
  jacobian: ROPJacobian;
  t0: number;  // Reference time (spud)
  d0: number;  // Reference depth (surface)
}

/**
 * Metric tensor field for drilling manifold geometry
 *
 * At each point (t, d), the metric is:
 *     g = | 1      1/ROP |
 *         | 1/ROP  1     |
 *
 * This defines the non-Euclidean geometry where ROP warps space.
 */
export interface DrillingMetricTensor {
  g_tt: number;  // Time-time component (usually 1)
  g_dd: number;  // Depth-depth component (usually 1)
  g_td: number;  // Off-diagonal (1/ROP) - the warping term
  determinant: number;  // det(g) = 1 - 1/ROP²
  is_degenerate: boolean;  // True when ROP → 0 or ∞
}

/**
 * Geodesic on the drilling manifold
 *
 * The optimal path between two drilling states, accounting for
 * ROP-warped geometry. Minimizes the action S = ∫√(g_ij dx^i dx^j)
 */
export interface DrillingGeodesic {
  start: { t: number; d: number };
  end: { t: number; d: number };
  path: { t: number; d: number }[];
  total_length: number;
  curvature_integral: number;
}

/**
 * Curvature information at a point on the drilling manifold
 */
export interface ManifoldCurvature {
  ricci_scalar: number;  // R > 0: converging, R < 0: diverging
  gaussian_curvature: number;  // For 2D, K = R/2
  is_flat: boolean;  // True when |R| < threshold
}

/**
 * Enhanced drilling record with dual coordinates
 */
export interface DualCoordinateDrillingRecord extends DrillingRecord {
  time: number;  // Seconds from spud (reconstructed if needed)
  coordinate_system: CoordinateSystem;
  metric_tensor: DrillingMetricTensor;
}

// =============================================================================
// Utility Types
// =============================================================================

export type Vector = number[];
export type PointCloud = Vector[];
export type Tensor = number[] | number[][] | number[][][];

/**
 * Metric tensor type for ActivityState geometries
 */
export interface MetricTensor {
  type: 'euclidean' | 'warped' | 'custom';
  dimension: number;
  compute_distance: (p1: Vector, p2: Vector) => number;
  compute_geodesic: (start: Vector, end: Vector, num_points?: number) => Vector[];
}

/**
 * Component registration for Manifold Bridge
 */
export interface ComponentNode {
  name: string;
  module_path: string;
  connections: Record<string, 'EXTENDS' | 'USES' | 'TRANSFORMS' | 'COMPOSES' | 'IMPLEMENTS' | 'BRIDGES'>;
  metadata: Record<string, unknown>;
}

// =============================================================================
// Re-exports for convenience
// =============================================================================

export {
  ConditionState as State,
  ActivityState as Regime,
  ClassificationResult as Classification,
};
