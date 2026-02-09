"""
Point Cloud REST and WebSocket API Routes

Provides REST endpoints for point cloud operations:
- Format conversion with streaming support
- Spatial/temporal queries
- Trajectory queries
- Validation
- Manual override submission

WebSocket endpoint for:
- Real-time streaming data
- Override request/response flow
- Progress updates during conversion
"""

from __future__ import annotations
import asyncio
import json
import logging
import tempfile
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field

# Try importing point cloud modules
try:
    from jones_framework.data.point_cloud.conversion import (
        PointCloudConverter, ConversionOptions, ConversionResult,
        ConversionFormat, TimeEpoch, ConversionProgress,
        convert, convert_to_split_brain, detect_format
    )
    from jones_framework.data.point_cloud.validation import (
        ValidationChain, ValidationResult, ValidationError
    )
    from jones_framework.data.point_cloud.override import (
        OverrideManager, OverrideContext, OverrideDecision
    )
    from jones_framework.data.point_cloud.fast_point_cloud import FastPointCloud
    from jones_framework.data.point_cloud.split_brain import SplitBrainStore
    from jones_framework.data.point_cloud.stgi_index import STGIIndex
    from jones_framework.data.point_cloud.streaming import StreamingPointCloudLoader
    POINTCLOUD_AVAILABLE = True
except ImportError as e:
    POINTCLOUD_AVAILABLE = False
    _IMPORT_ERROR = str(e)

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models for Request/Response
# =============================================================================

class ConversionRequest(BaseModel):
    """Request model for point cloud conversion."""
    source_path: str = Field(..., description="Path to source point cloud file")
    target_path: str = Field(..., description="Path for output file")
    source_format: Optional[str] = Field(None, description="Source format (auto-detect if None)")
    target_format: Optional[str] = Field(None, description="Target format (infer from extension if None)")
    streaming: bool = Field(True, description="Use streaming conversion")
    validate: bool = Field(True, description="Validate during conversion")
    normalize_time: bool = Field(True, description="Normalize time to GPS epoch")
    source_epoch: str = Field("UNIX", description="Source time epoch")
    target_epoch: str = Field("GPS_ADJUSTED", description="Target time epoch")


class SplitBrainConversionRequest(BaseModel):
    """Request model for split-brain conversion."""
    source_path: str = Field(..., description="Path to source point cloud file")
    output_dir: str = Field(..., description="Output directory")
    name: str = Field("pointcloud", description="Base name for output files")
    validate: bool = Field(True, description="Validate during conversion")
    normalize_time: bool = Field(True, description="Normalize time to GPS epoch")


class SpatialQueryRequest(BaseModel):
    """Request model for spatial queries."""
    center: List[float] = Field(..., description="Query center point [X, Y, Z]")
    radius: Optional[float] = Field(None, description="Search radius")
    k_nearest: Optional[int] = Field(None, description="Number of nearest neighbors")
    store_path: Optional[str] = Field(None, description="Path to split-brain store manifest")


class TemporalQueryRequest(BaseModel):
    """Request model for temporal queries."""
    t_start: float = Field(..., description="Start time (GPS seconds)")
    t_end: float = Field(..., description="End time (GPS seconds)")
    store_path: Optional[str] = Field(None, description="Path to split-brain store manifest")


class SpatioTemporalQueryRequest(BaseModel):
    """Request model for combined spatio-temporal queries."""
    center: List[float] = Field(..., description="Query center point [X, Y, Z]")
    radius: float = Field(..., description="Spatial search radius")
    t_start: float = Field(..., description="Start time (GPS seconds)")
    t_end: float = Field(..., description="End time (GPS seconds)")
    store_path: Optional[str] = Field(None, description="Path to split-brain store manifest")


class TrajectoryQueryRequest(BaseModel):
    """Request model for trajectory queries."""
    waypoints: List[List[float]] = Field(..., description="List of [X, Y, Z] waypoints")
    radii: List[float] = Field(..., description="Search radius at each waypoint")
    time_windows: List[List[float]] = Field(..., description="List of [t_start, t_end] windows")
    store_path: Optional[str] = Field(None, description="Path to split-brain store manifest")


class ValidationRequest(BaseModel):
    """Request model for validation."""
    source_path: Optional[str] = Field(None, description="Path to point cloud file")
    point_cloud: Optional[List[List[float]]] = Field(None, description="Inline point cloud data")
    auto_fix: bool = Field(False, description="Attempt to auto-fix issues")
    validators: Optional[List[str]] = Field(None, description="Specific validators to run")


class OverrideSubmitRequest(BaseModel):
    """Request model for submitting override decision."""
    context_id: str = Field(..., description="Override context ID")
    action: str = Field(..., description="Action: 'accept', 'reject', or 'modify'")
    new_value: Optional[Dict[str, Any]] = Field(None, description="New value if action is 'modify'")
    reason: Optional[str] = Field(None, description="Reason for decision")


class ConversionResponse(BaseModel):
    """Response model for conversion operations."""
    success: bool
    job_id: Optional[str] = None
    source_path: Optional[str] = None
    target_path: Optional[str] = None
    geometry_path: Optional[str] = None
    temporal_path: Optional[str] = None
    manifest_path: Optional[str] = None
    total_points: int = 0
    elapsed_seconds: float = 0.0
    checksum: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    """Response model for query operations."""
    success: bool
    num_results: int = 0
    indices: List[int] = Field(default_factory=list)
    points: Optional[List[List[float]]] = None
    times: Optional[List[float]] = None
    elapsed_ms: float = 0.0
    error: Optional[str] = None


class ValidationResponse(BaseModel):
    """Response model for validation."""
    is_valid: bool
    total_errors: int = 0
    total_warnings: int = 0
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    auto_fixed: int = 0
    elapsed_ms: float = 0.0


class ProgressResponse(BaseModel):
    """Response model for progress updates."""
    job_id: str
    current_stage: str
    percent_complete: float
    processed_points: int
    total_points: int
    points_per_second: float
    elapsed_seconds: float
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


# =============================================================================
# Application State
# =============================================================================

class PointCloudState:
    """Shared state for point cloud operations."""

    def __init__(self):
        self.converter: Optional[PointCloudConverter] = None
        self.validation_chain: Optional[ValidationChain] = None
        self.override_manager: Optional[OverrideManager] = None
        self.active_stores: Dict[str, SplitBrainStore] = {}
        self.active_jobs: Dict[str, ConversionProgress] = {}
        self.pending_overrides: Dict[str, OverrideContext] = {}
        self.websocket_connections: List[WebSocket] = []
        self._job_counter = 0

    def initialize(self):
        """Initialize point cloud components."""
        if not POINTCLOUD_AVAILABLE:
            logger.warning(f"Point cloud modules not available: {_IMPORT_ERROR}")
            return

        self.converter = PointCloudConverter(
            progress_callback=self._on_progress
        )
        self.validation_chain = ValidationChain.default_chain()
        self.override_manager = OverrideManager()

        logger.info("Point cloud API initialized")

    def _on_progress(self, progress: ConversionProgress):
        """Callback for conversion progress."""
        # Will be used to broadcast to WebSocket clients
        pass

    def generate_job_id(self) -> str:
        """Generate unique job ID."""
        self._job_counter += 1
        return f"pc_job_{self._job_counter}_{int(time.time())}"


# Global state instance
pc_state = PointCloudState()


# =============================================================================
# Router Setup
# =============================================================================

router = APIRouter(prefix='/api/v1/pointcloud', tags=['Point Cloud'])


def _check_available():
    """Check if point cloud modules are available."""
    if not POINTCLOUD_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=f"Point cloud modules not available: {_IMPORT_ERROR}"
        )


# =============================================================================
# REST Endpoints
# =============================================================================

@router.get('/status')
async def get_status():
    """Get point cloud API status."""
    return {
        'available': POINTCLOUD_AVAILABLE,
        'converter_ready': pc_state.converter is not None,
        'validation_ready': pc_state.validation_chain is not None,
        'active_stores': len(pc_state.active_stores),
        'active_jobs': len(pc_state.active_jobs),
        'supported_formats': [f.name for f in ConversionFormat] if POINTCLOUD_AVAILABLE else []
    }


@router.post('/convert', response_model=ConversionResponse)
async def convert_pointcloud(request: ConversionRequest, background_tasks: BackgroundTasks):
    """
    Convert point cloud between formats.

    Supports streaming conversion for large files.
    Returns job ID for progress tracking.
    """
    _check_available()

    job_id = pc_state.generate_job_id()

    # Parse options
    try:
        source_fmt = ConversionFormat[request.source_format.upper()] if request.source_format else None
        target_fmt = ConversionFormat[request.target_format.upper()] if request.target_format else None
        source_epoch = TimeEpoch[request.source_epoch.upper()]
        target_epoch = TimeEpoch[request.target_epoch.upper()]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Invalid format/epoch: {e}")

    options = ConversionOptions(
        source_format=source_fmt,
        target_format=target_fmt,
        streaming=request.streaming,
        validate=request.validate,
        normalize_time=request.normalize_time,
        source_epoch=source_epoch,
        target_epoch=target_epoch
    )

    # Run conversion
    start_time = time.time()
    result = pc_state.converter.convert(
        request.source_path,
        request.target_path,
        options
    )

    return ConversionResponse(
        success=result.success,
        job_id=job_id,
        source_path=result.source_path,
        target_path=result.target_path,
        total_points=result.total_points,
        elapsed_seconds=result.elapsed_seconds,
        checksum=result.checksum,
        errors=result.errors,
        warnings=result.warnings
    )


@router.post('/convert/split-brain', response_model=ConversionResponse)
async def convert_to_splitbrain(request: SplitBrainConversionRequest):
    """
    Convert to split-brain storage (COPC geometry + Parquet temporal).

    Creates three files:
    - {name}.copc.laz - Geometry (COPC format)
    - {name}_temporal.parquet - Temporal index
    - {name}_manifest.json - Manifest linking both
    """
    _check_available()

    options = ConversionOptions(
        validate=request.validate,
        normalize_time=request.normalize_time
    )

    output_dir = Path(request.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    geometry_path = output_dir / f"{request.name}.copc.laz"
    temporal_path = output_dir / f"{request.name}_temporal.parquet"
    manifest_path = output_dir / f"{request.name}_manifest.json"

    result = pc_state.converter.convert_to_split_brain(
        request.source_path,
        geometry_path,
        temporal_path,
        manifest_path,
        options
    )

    return ConversionResponse(
        success=result.success,
        source_path=result.source_path,
        geometry_path=result.geometry_path,
        temporal_path=result.temporal_path,
        manifest_path=result.manifest_path,
        total_points=result.total_points,
        elapsed_seconds=result.elapsed_seconds,
        checksum=result.checksum,
        errors=result.errors,
        warnings=result.warnings
    )


@router.post('/convert/stream')
async def stream_convert(request: ConversionRequest):
    """
    Stream conversion progress via Server-Sent Events (SSE).

    Returns progress updates as the conversion proceeds.
    """
    _check_available()

    async def generate():
        try:
            source_fmt = ConversionFormat[request.source_format.upper()] if request.source_format else None
            target_fmt = ConversionFormat[request.target_format.upper()] if request.target_format else None
        except KeyError as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        options = ConversionOptions(
            source_format=source_fmt,
            target_format=target_fmt,
            streaming=True,
            validate=request.validate
        )

        try:
            for progress in pc_state.converter.stream_convert(
                request.source_path,
                request.target_path,
                options
            ):
                yield f"data: {json.dumps({
                    'stage': progress.current_stage,
                    'percent': progress.percent_complete,
                    'processed': progress.processed_points,
                    'total': progress.total_points,
                    'rate': progress.points_per_second
                })}\n\n"
                await asyncio.sleep(0.01)  # Yield control

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type='text/event-stream'
    )


@router.post('/query/spatial', response_model=QueryResponse)
async def query_spatial(request: SpatialQueryRequest):
    """
    Query points by spatial location.

    Supports radius search or k-nearest neighbors.
    """
    _check_available()

    start = time.time()

    # Get or load store
    store = None
    if request.store_path:
        if request.store_path in pc_state.active_stores:
            store = pc_state.active_stores[request.store_path]
        else:
            try:
                store = SplitBrainStore.from_manifest(request.store_path)
                pc_state.active_stores[request.store_path] = store
            except Exception as e:
                return QueryResponse(success=False, error=str(e))

    if store is None:
        return QueryResponse(success=False, error="No store specified or loaded")

    center = np.array(request.center, dtype=np.float32)

    try:
        if request.radius is not None:
            indices = store.geometry_store.query_radius(center, request.radius)
        elif request.k_nearest is not None:
            indices, _ = store.geometry_store.query_knn(center, request.k_nearest)
        else:
            return QueryResponse(success=False, error="Must specify radius or k_nearest")

        elapsed_ms = (time.time() - start) * 1000

        return QueryResponse(
            success=True,
            num_results=len(indices),
            indices=indices.tolist(),
            elapsed_ms=elapsed_ms
        )

    except Exception as e:
        return QueryResponse(success=False, error=str(e))


@router.post('/query/temporal', response_model=QueryResponse)
async def query_temporal(request: TemporalQueryRequest):
    """Query points by time range."""
    _check_available()

    start = time.time()

    store = None
    if request.store_path:
        if request.store_path in pc_state.active_stores:
            store = pc_state.active_stores[request.store_path]
        else:
            try:
                store = SplitBrainStore.from_manifest(request.store_path)
                pc_state.active_stores[request.store_path] = store
            except Exception as e:
                return QueryResponse(success=False, error=str(e))

    if store is None:
        return QueryResponse(success=False, error="No store specified or loaded")

    try:
        indices = store.temporal_store.query_time_range(request.t_start, request.t_end)
        elapsed_ms = (time.time() - start) * 1000

        return QueryResponse(
            success=True,
            num_results=len(indices),
            indices=indices.tolist(),
            elapsed_ms=elapsed_ms
        )

    except Exception as e:
        return QueryResponse(success=False, error=str(e))


@router.post('/query/spatiotemporal', response_model=QueryResponse)
async def query_spatiotemporal(request: SpatioTemporalQueryRequest):
    """Query points by spatial location AND time range."""
    _check_available()

    start = time.time()

    store = None
    if request.store_path:
        if request.store_path in pc_state.active_stores:
            store = pc_state.active_stores[request.store_path]
        else:
            try:
                store = SplitBrainStore.from_manifest(request.store_path)
                pc_state.active_stores[request.store_path] = store
            except Exception as e:
                return QueryResponse(success=False, error=str(e))

    if store is None:
        return QueryResponse(success=False, error="No store specified or loaded")

    center = np.array(request.center, dtype=np.float32)

    try:
        # Get spatial hits
        spatial_indices = set(store.geometry_store.query_radius(center, request.radius))

        # Get temporal hits
        temporal_indices = set(store.temporal_store.query_time_range(
            request.t_start, request.t_end
        ))

        # Intersection
        indices = list(spatial_indices & temporal_indices)
        elapsed_ms = (time.time() - start) * 1000

        return QueryResponse(
            success=True,
            num_results=len(indices),
            indices=indices,
            elapsed_ms=elapsed_ms
        )

    except Exception as e:
        return QueryResponse(success=False, error=str(e))


@router.post('/query/trajectory', response_model=QueryResponse)
async def query_trajectory(request: TrajectoryQueryRequest):
    """Query points along a trajectory with time windows."""
    _check_available()

    start = time.time()

    if len(request.waypoints) != len(request.radii) != len(request.time_windows):
        return QueryResponse(
            success=False,
            error="waypoints, radii, and time_windows must have same length"
        )

    store = None
    if request.store_path:
        if request.store_path in pc_state.active_stores:
            store = pc_state.active_stores[request.store_path]
        else:
            try:
                store = SplitBrainStore.from_manifest(request.store_path)
                pc_state.active_stores[request.store_path] = store
            except Exception as e:
                return QueryResponse(success=False, error=str(e))

    if store is None:
        return QueryResponse(success=False, error="No store specified or loaded")

    try:
        all_indices = set()

        for waypoint, radius, (t_start, t_end) in zip(
            request.waypoints, request.radii, request.time_windows
        ):
            center = np.array(waypoint, dtype=np.float32)

            # Spatial query
            spatial_indices = set(store.geometry_store.query_radius(center, radius))

            # Temporal query
            temporal_indices = set(store.temporal_store.query_time_range(t_start, t_end))

            # Add intersection
            all_indices.update(spatial_indices & temporal_indices)

        elapsed_ms = (time.time() - start) * 1000

        return QueryResponse(
            success=True,
            num_results=len(all_indices),
            indices=list(all_indices),
            elapsed_ms=elapsed_ms
        )

    except Exception as e:
        return QueryResponse(success=False, error=str(e))


@router.post('/validate', response_model=ValidationResponse)
async def validate_pointcloud(request: ValidationRequest):
    """
    Validate point cloud data.

    Can validate from file path or inline data.
    Returns detailed error information.
    """
    _check_available()

    start = time.time()

    # Get data
    if request.source_path:
        try:
            # Load file
            fmt = detect_format(request.source_path)
            loader = StreamingPointCloudLoader(
                source=request.source_path,
                chunk_size=100_000
            )
            # Load all for validation
            chunks = list(loader.stream())
            data = np.vstack(chunks) if chunks else np.array([])
        except Exception as e:
            return ValidationResponse(
                is_valid=False,
                total_errors=1,
                errors=[{'message': str(e), 'validator': 'loader'}]
            )
    elif request.point_cloud:
        data = np.array(request.point_cloud)
    else:
        return ValidationResponse(
            is_valid=False,
            total_errors=1,
            errors=[{'message': 'Must provide source_path or point_cloud'}]
        )

    # Run validation
    chain = pc_state.validation_chain
    if request.validators:
        # Filter to specific validators
        chain = ValidationChain(validators=[
            v for v in chain.validators
            if v.__class__.__name__ in request.validators
        ])

    result = chain.validate(data, auto_fix=request.auto_fix)
    elapsed_ms = (time.time() - start) * 1000

    return ValidationResponse(
        is_valid=result.is_valid,
        total_errors=len(result.errors),
        total_warnings=len(result.warnings),
        errors=[
            {
                'validator': e.validator_name,
                'message': e.message,
                'field': e.field_name,
                'indices': e.affected_indices[:100] if e.affected_indices else []
            }
            for e in result.errors
        ],
        auto_fixed=len(result.auto_fixed) if hasattr(result, 'auto_fixed') else 0,
        elapsed_ms=elapsed_ms
    )


@router.post('/override')
async def submit_override(request: OverrideSubmitRequest):
    """
    Submit a manual override decision.

    Used when conversion/validation requests user intervention.
    """
    _check_available()

    if request.context_id not in pc_state.pending_overrides:
        raise HTTPException(status_code=404, detail="Override context not found")

    context = pc_state.pending_overrides.pop(request.context_id)

    decision = OverrideDecision(
        action=request.action,
        new_value=request.new_value,
        reason=request.reason,
        user_id='api_user',
        timestamp=time.time()
    )

    # Apply decision through override manager
    if pc_state.override_manager:
        pc_state.override_manager.record_decision(context, decision)

    return {
        'success': True,
        'context_id': request.context_id,
        'action': request.action
    }


@router.get('/job/{job_id}', response_model=ProgressResponse)
async def get_job_status(job_id: str):
    """Get status of a conversion job."""
    _check_available()

    if job_id not in pc_state.active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    progress = pc_state.active_jobs[job_id]

    return ProgressResponse(
        job_id=job_id,
        current_stage=progress.current_stage,
        percent_complete=progress.percent_complete,
        processed_points=progress.processed_points,
        total_points=progress.total_points,
        points_per_second=progress.points_per_second,
        elapsed_seconds=progress.elapsed_seconds,
        errors=progress.errors,
        warnings=progress.warnings
    )


@router.get('/formats')
async def list_formats():
    """List supported point cloud formats."""
    if not POINTCLOUD_AVAILABLE:
        return {'formats': [], 'epochs': []}

    return {
        'formats': [
            {
                'name': f.name,
                'extensions': _get_format_extensions(f)
            }
            for f in ConversionFormat
        ],
        'epochs': [e.name for e in TimeEpoch]
    }


def _get_format_extensions(fmt: ConversionFormat) -> List[str]:
    """Get file extensions for a format."""
    mapping = {
        ConversionFormat.LAS: ['.las'],
        ConversionFormat.LAZ: ['.laz'],
        ConversionFormat.NUMPY: ['.npy', '.npz'],
        ConversionFormat.CSV: ['.csv'],
        ConversionFormat.PARQUET: ['.parquet'],
        ConversionFormat.HDF5: ['.h5', '.hdf5'],
        ConversionFormat.COPC: ['.copc.laz'],
        ConversionFormat.SPLIT_BRAIN: ['.copc.laz', '.parquet'],
        ConversionFormat.PLY: ['.ply'],
        ConversionFormat.PCD: ['.pcd'],
    }
    return mapping.get(fmt, [])


# =============================================================================
# WebSocket Endpoint
# =============================================================================

@router.websocket('/ws')
async def pointcloud_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for point cloud operations.

    Message types:
    - subscribe: Subscribe to store updates
    - data: Stream point chunks (binary)
    - override_request: Request user override decision
    - override_response: User's override decision
    - progress: Conversion progress updates
    """
    await websocket.accept()
    pc_state.websocket_connections.append(websocket)

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get('type')

            if msg_type == 'ping':
                await websocket.send_json({'type': 'pong'})

            elif msg_type == 'subscribe':
                # Subscribe to store updates
                store_path = data.get('store_path')
                if store_path and POINTCLOUD_AVAILABLE:
                    try:
                        if store_path not in pc_state.active_stores:
                            store = SplitBrainStore.from_manifest(store_path)
                            pc_state.active_stores[store_path] = store
                        await websocket.send_json({
                            'type': 'subscribed',
                            'store_path': store_path
                        })
                    except Exception as e:
                        await websocket.send_json({
                            'type': 'error',
                            'message': str(e)
                        })

            elif msg_type == 'stream_start':
                # Start streaming conversion
                source = data.get('source_path')
                target = data.get('target_path')

                if source and target and POINTCLOUD_AVAILABLE:
                    job_id = pc_state.generate_job_id()

                    async def stream_job():
                        options = ConversionOptions(streaming=True)
                        for progress in pc_state.converter.stream_convert(
                            source, target, options
                        ):
                            await websocket.send_json({
                                'type': 'progress',
                                'job_id': job_id,
                                'stage': progress.current_stage,
                                'percent': progress.percent_complete,
                                'processed': progress.processed_points
                            })
                            await asyncio.sleep(0.01)

                        await websocket.send_json({
                            'type': 'complete',
                            'job_id': job_id
                        })

                    asyncio.create_task(stream_job())

                    await websocket.send_json({
                        'type': 'stream_started',
                        'job_id': job_id
                    })

            elif msg_type == 'override_response':
                # User responding to override request
                context_id = data.get('context_id')
                action = data.get('action', 'accept')
                new_value = data.get('new_value')

                if context_id in pc_state.pending_overrides:
                    context = pc_state.pending_overrides.pop(context_id)

                    decision = OverrideDecision(
                        action=action,
                        new_value=new_value,
                        user_id='ws_user',
                        timestamp=time.time()
                    )

                    if pc_state.override_manager:
                        pc_state.override_manager.record_decision(context, decision)

                    await websocket.send_json({
                        'type': 'override_applied',
                        'context_id': context_id
                    })

            elif msg_type == 'query':
                # Real-time query
                query_type = data.get('query_type')
                store_path = data.get('store_path')

                if store_path in pc_state.active_stores:
                    store = pc_state.active_stores[store_path]

                    if query_type == 'spatial':
                        center = np.array(data.get('center'), dtype=np.float32)
                        radius = data.get('radius')
                        indices = store.geometry_store.query_radius(center, radius)

                        await websocket.send_json({
                            'type': 'query_result',
                            'query_type': 'spatial',
                            'indices': indices.tolist(),
                            'count': len(indices)
                        })

                    elif query_type == 'temporal':
                        t_start = data.get('t_start')
                        t_end = data.get('t_end')
                        indices = store.temporal_store.query_time_range(t_start, t_end)

                        await websocket.send_json({
                            'type': 'query_result',
                            'query_type': 'temporal',
                            'indices': indices.tolist(),
                            'count': len(indices)
                        })

    except WebSocketDisconnect:
        pc_state.websocket_connections.remove(websocket)
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
        try:
            await websocket.send_json({'type': 'error', 'message': str(e)})
        except:
            pass
        pc_state.websocket_connections.remove(websocket)


# =============================================================================
# Broadcast Helpers
# =============================================================================

async def broadcast_override_request(context: OverrideContext):
    """Broadcast override request to all connected clients."""
    context_id = f"override_{int(time.time())}_{id(context)}"
    pc_state.pending_overrides[context_id] = context

    message = {
        'type': 'override_request',
        'context_id': context_id,
        'hook_name': context.hook_name,
        'description': context.description,
        'current_value': context.current_value,
        'timestamp': datetime.utcnow().isoformat()
    }

    for ws in pc_state.websocket_connections:
        try:
            await ws.send_json(message)
        except:
            pass


async def broadcast_progress(job_id: str, progress: ConversionProgress):
    """Broadcast conversion progress to all connected clients."""
    message = {
        'type': 'progress',
        'job_id': job_id,
        'stage': progress.current_stage,
        'percent': progress.percent_complete,
        'processed': progress.processed_points,
        'total': progress.total_points,
        'rate': progress.points_per_second
    }

    for ws in pc_state.websocket_connections:
        try:
            await ws.send_json(message)
        except:
            pass


# =============================================================================
# Initialization Function
# =============================================================================

def initialize_pointcloud_routes():
    """Initialize point cloud state on application startup."""
    pc_state.initialize()


def get_routes():
    """Get all point cloud routes."""
    return router
