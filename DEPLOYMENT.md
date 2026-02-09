# Enterprise Deployment Guide

## Target Hardware: 8-12GB VRAM GPUs

This guide covers deploying the Unified Activity:State Platform on enterprise
hardware with limited GPU memory (RTX 3080, RTX 4070, A4000, etc.).

## Hardware Requirements

### Minimum Configuration
- **GPU**: 8GB VRAM (RTX 3080, RTX 4070, A4000)
- **CPU**: 8 cores
- **RAM**: 32GB
- **Storage**: 50GB SSD

### Recommended Configuration
- **GPU**: 12GB VRAM (RTX 4070 Ti, A4500)
- **CPU**: 16 cores
- **RAM**: 64GB
- **Storage**: 100GB NVMe SSD

### Air-Gapped Deployment
For offline/air-gapped environments:
- All model weights must be pre-downloaded
- No internet connectivity required after initial setup
- Local model registry supported

## Memory Optimization Strategies

### 1. Quantization

Reduce model memory footprint with weight quantization:

```python
from jones_framework.ml.optimization import (
    optimize_for_vram,
    QuantizationType,
    QuantizationConfig,
)

# Get recommended settings for your hardware
config = optimize_for_vram(
    model_params=100_000_000,  # 100M parameter model
    vram_preset="rtx_4070_12gb",
    target_batch_size=32
)

print(config)
# {
#   'recommended_quantization': 'INT8',
#   'fits_in_vram': True,
#   'max_batch_size': 64,
#   'gradient_checkpointing': False,
# }
```

### Memory Savings by Quantization Type

| Quantization | Bits | Compression | 100M Model | Quality Impact |
|--------------|------|-------------|------------|----------------|
| FP32         | 32   | 1x          | 400 MB     | Baseline       |
| FP16         | 16   | 2x          | 200 MB     | Minimal        |
| INT8         | 8    | 4x          | 100 MB     | Low            |
| INT4         | 4    | 8x          | 50 MB      | Moderate       |
| NF4 (QLoRA)  | 4    | 8x          | 50 MB      | Low            |

### 2. Expert Offloading (MoE)

For Mixture of Experts models, only active experts stay on GPU:

```python
from jones_framework.ml.optimization import ExpertOffloader

offloader = ExpertOffloader(
    gpu_budget_bytes=8 * 1024**3,  # 8GB
    experts_per_gpu=2
)

# Register all experts (stored on CPU)
for expert_id, weights in expert_weights.items():
    offloader.register_expert(expert_id, weights)

# Load needed expert on-demand
active_weights = offloader.ensure_on_gpu("drilling_expert_1")
```

### 3. Dynamic Batching

Maximize throughput while respecting memory limits:

```python
from jones_framework.ml.optimization import DynamicBatcher

batcher = DynamicBatcher(
    max_batch_size=32,
    max_tokens_per_batch=8192,
    timeout_ms=100.0
)

# Add requests as they arrive
batcher.add_request("req_1", sequence_1)
batcher.add_request("req_2", sequence_2)

# Process when batch is ready
if batcher.should_flush():
    batch = batcher.get_batch()
    outputs = model(batch["sequences"], batch["attention_mask"])
```

### 4. Chunked Sequence Processing

Handle long sequences that exceed GPU memory:

```python
from jones_framework.ml.optimization import ChunkedProcessor

chunker = ChunkedProcessor(
    chunk_size=512,
    overlap=64,
    aggregation="mean"
)

# Process long sequence in chunks
for chunk, start, end in chunker.chunk_sequence(long_sequence):
    output = model(chunk)
    chunk_outputs.append(output)

# Aggregate results
final_output = chunker.aggregate_outputs(chunk_outputs, positions, seq_len)
```

## Deployment Configurations

### RTX 3080 (10GB) - Minimum Viable

```python
from jones_framework.ml.optimization import INFERENCE_PRESETS

config = INFERENCE_PRESETS["rtx_3080_10gb"]
# max_memory_gb: 10.0
# max_batch_size: 16
# chunk_size: 512
# quantization: INT8 recommended
# experts_on_gpu: 2
```

### RTX 4070 (12GB) - Standard Enterprise

```python
config = INFERENCE_PRESETS["rtx_4070_12gb"]
# max_memory_gb: 12.0
# max_batch_size: 24
# chunk_size: 768
# quantization: INT8 or FP16
# experts_on_gpu: 3
```

### RTX 4080 (16GB) - High Performance

```python
config = INFERENCE_PRESETS["rtx_4080_16gb"]
# max_memory_gb: 16.0
# max_batch_size: 32
# chunk_size: 1024
# quantization: FP16
# experts_on_gpu: 4
```

### CPU-Only - Air-Gapped/Secure

```python
config = INFERENCE_PRESETS["cpu_only"]
# No GPU required
# Slower but works anywhere
# Consider INT8 quantization for speed
```

## Quick Start

### Option A: Using the CLI (Recommended)

```bash
cd unified-activity-state-platform/backend

# Install with CLI
pip install -e ".[cli,api]"

# Run the diagnostic tool to verify your setup
jones doctor

# Start services
jones start
```

### Option B: Manual Installation

```bash
cd unified-activity-state-platform

# Backend
pip install -e backend[api]

# Frontend (optional)
cd frontend && pnpm install
```

### 2. Configure for Your Hardware

```bash
export JONES_VRAM_PRESET="rtx_4070_12gb"
export JONES_MAX_BATCH_SIZE=24
export JONES_QUANTIZATION="INT8"
```

### 3. Start API Server

```bash
# Development
uvicorn jones_framework.api.server:app --reload --port 8000

# Production
gunicorn jones_framework.api.server:app -w 4 -k uvicorn.workers.UvicornWorker
```

### 4. Health Check

```bash
curl http://localhost:8000/health
# {"status": "healthy", "memory_usage": "4.2 GB / 12.0 GB"}
```

## Monitoring

### Memory Dashboard

The API exposes memory metrics at `/metrics`:

```json
{
  "gpu_memory_used_gb": 4.2,
  "gpu_memory_total_gb": 12.0,
  "model_memory_gb": 2.1,
  "kv_cache_mb": 512,
  "active_experts": 2,
  "batch_queue_size": 5
}
```

### Alerts

Configure alerts for:
- GPU memory > 90% utilization
- Batch queue > 100 requests
- Inference latency > 500ms p99

## Troubleshooting

Use the built-in diagnostic tool for quick troubleshooting:

```bash
# Run full diagnostics
jones doctor --verbose

# Attempt automatic fixes
jones doctor --fix

# View service logs
jones logs --follow
```

### Out of Memory (OOM)

1. Reduce `max_batch_size`
2. Enable more aggressive quantization (INT4)
3. Reduce `chunk_size` for long sequences
4. Enable expert offloading

### Slow Inference

1. Increase `max_batch_size` (if memory allows)
2. Enable KV caching
3. Prefetch likely experts
4. Use FP16 instead of INT8 (faster matmuls)

### Model Loading Fails

1. Check available disk space
2. Verify model file integrity
3. Try memory-mapped loading
4. Increase system RAM swap

### Services Won't Start

1. Run `jones doctor` to diagnose issues
2. Check port availability with `jones status`
3. Verify Python/Node versions
4. Check virtual environment: `jones install --check`

## Security Considerations

- Models run locally, no data leaves the device
- Supports air-gapped deployment
- API authentication via JWT tokens
- Audit logging for all predictions

## Support Matrix

| GPU | VRAM | Max Model Size | Recommended Quantization |
|-----|------|----------------|--------------------------|
| RTX 3080 | 10GB | 500M params | INT8 |
| RTX 4070 | 12GB | 700M params | INT8/FP16 |
| RTX 4080 | 16GB | 1B params | FP16 |
| RTX 3090 | 24GB | 2B params | FP16 |
| A4000 | 16GB | 1B params | FP16 |
| A5000 | 24GB | 2B params | FP16 |
| CPU Only | N/A | 500M params | INT8 |
