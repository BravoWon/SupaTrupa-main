/**
 * Auto-generated TypeScript types from Python jones_framework
 * 
 * DO NOT EDIT MANUALLY - Run `python scripts/generate_types.py` to regenerate
 * Generated from 4 dataclasses and 1 enums
 */

// =============================================================================
// Enums
// =============================================================================

export type RegimeID = 'darcy_flow' | 'non_darcy_flow' | 'turbulent' | 'multiphase' | 'bit_bounce' | 'packoff' | 'optimal' | 'stick_slip' | 'whirl' | 'formation_change' | 'washout' | 'lost_circulation' | 'normal' | 'transition' | 'kick' | 'unknown';

// =============================================================================
// Interfaces
// =============================================================================

export interface ConditionState {
  timestamp: number;
  vector: number[];
  metadata?: Record<string, unknown>;
  verified?: boolean;
  state_id?: string;
}

export interface ActivityState {
  regime_id: RegimeID;
  manifold_metric: ManifoldMetric;
  expert_model?: ExpertModel | null;
  transition_threshold?: number;
  metadata?: Record<string, unknown>;
}

/**
 * Configuration for a registered expert.
 */
export interface ExpertConfig {
  regime_id: RegimeID;
  expert: Expert;
  adapter?: LoRAAdapter | null;
  priority?: number;
  description?: string;
}

/**
 * Single LoRA layer implementing low-rank decomposition.
 * 
 * W_adapted = W_base + alpha * (A @ B)
 * 
 * Where:
 * - A: (input_dim, rank) - learned down-projection
 * - B: (rank, output_dim) - learned up-projection
 * - alpha: scaling factor for adaptation strength
 * 
 * This allows O(rank * (input + output)) parameters instead of
 * O(input * output), enabling fast switching between regimes.
 */
export interface LoRALayer {
  input_dim: number;
  output_dim: number;
  rank?: number;
  alpha?: number;
  name?: string;
  A?: number[];
  B?: number[];
}
