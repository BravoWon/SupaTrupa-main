from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
from jones_framework.api.rest.server import Route, HTTPMethod, APIResponse
from jones_framework.domains.drilling import DrillingAdapter, DrillingConfig, DrillingRegime, DrillingConditionState, CoordinateSystem, WellTrajectory, NPTCategory, NPTEvent, PipeTally, create_standard_tally, create_drilling_adapter, UncertainValue
from jones_framework.ml.optimization import optimize_for_vram, create_engine_for_hardware, VRAM_PRESETS, INFERENCE_PRESETS

class _cI104Bl:

    def __init__(self):
        self._adapters: Dict[str, DrillingAdapter] = {}
        self._default_adapter: Optional[DrillingAdapter] = None

    def _fOO04B2(self) -> List[Route]:
        return [Route('/drilling/adapters', HTTPMethod.GET, self.list_adapters, 'List all drilling adapters', auth_required=True), Route('/drilling/adapters', HTTPMethod.POST, self.create_adapter, 'Create new drilling adapter', auth_required=True), Route('/drilling/adapters/{adapter_id}', HTTPMethod.GET, self.get_adapter, 'Get drilling adapter details', auth_required=True), Route('/drilling/adapters/{adapter_id}', HTTPMethod.DELETE, self.delete_adapter, 'Delete drilling adapter', auth_required=True), Route('/drilling/ingest', HTTPMethod.POST, self.ingest_data, 'Ingest drilling data', auth_required=True), Route('/drilling/ingest/batch', HTTPMethod.POST, self.ingest_batch, 'Ingest batch of drilling data', auth_required=True), Route('/drilling/regime', HTTPMethod.GET, self.get_regime, 'Get current drilling regime', auth_required=True), Route('/drilling/regime/history', HTTPMethod.GET, self.get_regime_history, 'Get drilling regime history', auth_required=True), Route('/drilling/coordinate/time-at-depth', HTTPMethod.POST, self.time_at_depth, 'Get time coordinate at depth', auth_required=True), Route('/drilling/coordinate/depth-at-time', HTTPMethod.POST, self.depth_at_time, 'Get depth coordinate at time', auth_required=True), Route('/drilling/coordinate/dual-transform', HTTPMethod.POST, self.dual_transform, 'Transform to dual coordinates', auth_required=True), Route('/drilling/trajectory', HTTPMethod.GET, self.get_trajectory, 'Get well trajectory summary', auth_required=True), Route('/drilling/trajectory/stands', HTTPMethod.GET, self.get_stands, 'Get stand details', auth_required=True), Route('/drilling/trajectory/boundary', HTTPMethod.POST, self.add_boundary, 'Add stand boundary', auth_required=True), Route('/drilling/trajectory/rop-profile', HTTPMethod.GET, self.get_rop_profile, 'Get ROP profile', auth_required=True), Route('/drilling/npt', HTTPMethod.GET, self.get_npt_summary, 'Get NPT summary', auth_required=True), Route('/drilling/npt', HTTPMethod.POST, self.add_npt_event, 'Add NPT event', auth_required=True), Route('/drilling/npt/breakdown', HTTPMethod.GET, self.get_npt_breakdown, 'Get NPT breakdown by category', auth_required=True), Route('/drilling/tally', HTTPMethod.GET, self.get_tally, 'Get pipe tally', auth_required=True), Route('/drilling/tally/validate', HTTPMethod.POST, self.validate_tally, 'Validate depth against tally', auth_required=True), Route('/drilling/uncertainty', HTTPMethod.GET, self.get_uncertainty_budget, 'Get uncertainty budget', auth_required=True), Route('/drilling/uncertainty/compute', HTTPMethod.POST, self.compute_uncertainty, 'Compute uncertainty for measurement', auth_required=True), Route('/drilling/metrics', HTTPMethod.GET, self.get_metrics, 'Get drilling metrics', auth_required=True), Route('/drilling/efficiency', HTTPMethod.GET, self.get_efficiency, 'Get drilling efficiency', auth_required=True), Route('/optimization/status', HTTPMethod.GET, self.get_optimization_status, 'Get optimization status', auth_required=True), Route('/optimization/recommend', HTTPMethod.POST, self.recommend_settings, 'Get recommended settings for hardware', auth_required=True)]

    def _fI1O4B3(self, _fOOO4B4: Optional[str]=None) -> DrillingAdapter:
        if _fOOO4B4 and _fOOO4B4 in self._adapters:
            return self._adapters[_fOOO4B4]
        if self._default_adapter is None:
            self._default_adapter = create_drilling_adapter('default')
            self._adapters['default'] = self._default_adapter
        return self._default_adapter

    async def _fl0O4B5(self, _fOlO4B6: Dict) -> APIResponse:
        adapters = [{'id': aid, 'name': a.drilling_config.name, 'metrics': a.get_metrics()} for aid, a in self._adapters.items()]
        return APIResponse(success=True, data={'adapters': adapters})

    async def _f1O04B7(self, _fOlO4B6: Dict) -> APIResponse:
        body = _fOlO4B6.get('body', {})
        name = body.get('name', f'well_{len(self._adapters) + 1}')
        _fOOO4B4 = body.get('id', name)
        config_kwargs = {k: v for k, v in body.items() if k not in ['name', 'id']}
        adapter = create_drilling_adapter(name=name, **config_kwargs)
        self._adapters[_fOOO4B4] = adapter
        if self._default_adapter is None:
            self._default_adapter = adapter
        return APIResponse(success=True, data={'id': _fOOO4B4, 'name': name, 'message': 'Adapter created successfully'})

    async def _f11I4B8(self, _fOlO4B6: Dict) -> APIResponse:
        _fOOO4B4 = _fOlO4B6.get('params', {}).get('adapter_id')
        if _fOOO4B4 not in self._adapters:
            return APIResponse(success=False, error=f'Adapter {_fOOO4B4} not found')
        adapter = self._adapters[_fOOO4B4]
        return APIResponse(success=True, data=adapter.to_dict())

    async def _f0l14B9(self, _fOlO4B6: Dict) -> APIResponse:
        _fOOO4B4 = _fOlO4B6.get('params', {}).get('adapter_id')
        if _fOOO4B4 not in self._adapters:
            return APIResponse(success=False, error=f'Adapter {_fOOO4B4} not found')
        del self._adapters[_fOOO4B4]
        return APIResponse(success=True, data={'message': f'Adapter {_fOOO4B4} deleted'})

    async def _fl104BA(self, _fOlO4B6: Dict) -> APIResponse:
        body = _fOlO4B6.get('body', {})
        _fOOO4B4 = body.get('adapter_id')
        data = body.get('data', {})
        adapter = self._fI1O4B3(_fOOO4B4)
        state = adapter.ingest(data)
        return APIResponse(success=True, data={'state_id': id(state), 'time': state.time, 'depth': state.depth, 'rop': state.rop})

    async def _fOl04BB(self, _fOlO4B6: Dict) -> APIResponse:
        body = _fOlO4B6.get('body', {})
        _fOOO4B4 = body.get('adapter_id')
        data_list = body.get('data', [])
        adapter = self._fI1O4B3(_fOOO4B4)
        states = adapter._fOl04BB(data_list)
        return APIResponse(success=True, data={'count': len(states), 'message': f'Ingested {len(states)} data points'})

    async def _fO0I4Bc(self, _fOlO4B6: Dict) -> APIResponse:
        params = _fOlO4B6.get('params', {})
        _fOOO4B4 = params.get('adapter_id')
        adapter = self._fI1O4B3(_fOOO4B4)
        regime, confidence = adapter.detect_drilling_regime()
        return APIResponse(success=True, data={'regime': regime.name, 'confidence': confidence, 'framework_regime': adapter._regime_mapper.to_framework_regime(regime).name})

    async def _fIII4Bd(self, _fOlO4B6: Dict) -> APIResponse:
        params = _fOlO4B6.get('params', {})
        _fOOO4B4 = params.get('adapter_id')
        adapter = self._fI1O4B3(_fOOO4B4)
        history = adapter._moe.get_transition_history()
        return APIResponse(success=True, data={'transitions': [{'from': t.from_regime.name, 'to': t.to_regime.name, 'timestamp': t.timestamp} for t in history]})

    async def _fOIl4BE(self, _fOlO4B6: Dict) -> APIResponse:
        body = _fOlO4B6.get('body', {})
        _fOOO4B4 = body.get('adapter_id')
        depth = body.get('depth', 0.0)
        adapter = self._fI1O4B3(_fOOO4B4)
        time_val = adapter.get_time_at_depth(depth)
        if time_val is None:
            return APIResponse(success=False, error='Could not compute time at depth')
        return APIResponse(success=True, data={'depth': depth, 'time': time_val, 'time_hours': time_val / 3600.0})

    async def _fIOO4Bf(self, _fOlO4B6: Dict) -> APIResponse:
        body = _fOlO4B6.get('body', {})
        _fOOO4B4 = body.get('adapter_id')
        time_val = body.get('time', 0.0)
        adapter = self._fI1O4B3(_fOOO4B4)
        depth = adapter.get_depth_at_time(time_val)
        if depth is None:
            return APIResponse(success=False, error='Could not compute depth at time')
        return APIResponse(success=True, data={'time': time_val, 'depth': depth})

    async def _f1Il4cO(self, _fOlO4B6: Dict) -> APIResponse:
        body = _fOlO4B6.get('body', {})
        _fOOO4B4 = body.get('adapter_id')
        depths = np.array(body.get('depths', []))
        features = np.array(body.get('features', []))
        adapter = self._fI1O4B3(_fOOO4B4)
        times, depths_out, features_out = adapter.to_dual_coordinates(depths, features)
        return APIResponse(success=True, data={'times': times.tolist(), 'depths': depths_out.tolist(), 'features': features_out.tolist() if features_out.size > 0 else []})

    async def _fO114cl(self, _fOlO4B6: Dict) -> APIResponse:
        params = _fOlO4B6.get('params', {})
        _fOOO4B4 = params.get('adapter_id')
        adapter = self._fI1O4B3(_fOOO4B4)
        summary = adapter.get_trajectory_summary()
        return APIResponse(success=True, data=summary)

    async def _f0114c2(self, _fOlO4B6: Dict) -> APIResponse:
        params = _fOlO4B6.get('params', {})
        _fOOO4B4 = params.get('adapter_id')
        adapter = self._fI1O4B3(_fOOO4B4)
        stands = [{'number': i + 1, 'start_depth': s.start_boundary.depth, 'end_depth': s.end_boundary.depth, 'start_time': s.start_boundary.time, 'end_time': s.end_boundary.time, 'length': s.stand_length, 'drilling_time': s.drilling_time, 'npt_time': s.npt_time, 'average_rop': s.average_rop, 'efficiency': s.drilling_efficiency} for i, s in enumerate(adapter._trajectory.stands)]
        return APIResponse(success=True, data={'stands': stands})

    async def _f0lO4c3(self, _fOlO4B6: Dict) -> APIResponse:
        body = _fOlO4B6.get('body', {})
        _fOOO4B4 = body.get('adapter_id')
        time_val = body.get('time', 0.0)
        depth = body.get('depth', 0.0)
        adapter = self._fI1O4B3(_fOOO4B4)
        adapter.add_stand_boundary(time=time_val, depth=depth)
        return APIResponse(success=True, data={'message': 'Stand boundary added', 'stands_count': len(adapter._trajectory.stands)})

    async def _f1II4c4(self, _fOlO4B6: Dict) -> APIResponse:
        params = _fOlO4B6.get('params', {})
        _fOOO4B4 = params.get('adapter_id')
        adapter = self._fI1O4B3(_fOOO4B4)
        depths, rops = adapter._f1II4c4()
        return APIResponse(success=True, data={'depths': depths.tolist(), 'rops': rops.tolist()})

    async def _flOl4c5(self, _fOlO4B6: Dict) -> APIResponse:
        params = _fOlO4B6.get('params', {})
        _fOOO4B4 = params.get('adapter_id')
        adapter = self._fI1O4B3(_fOOO4B4)
        summary = adapter._trajectory.summary()
        return APIResponse(success=True, data={'total_npt_time': summary.get('total_npt_time', 0), 'drilling_efficiency': summary.get('drilling_efficiency', 1.0)})

    async def _fl0l4c6(self, _fOlO4B6: Dict) -> APIResponse:
        body = _fOlO4B6.get('body', {})
        _fOOO4B4 = body.get('adapter_id')
        category = body.get('category', 'OTHER')
        start_time = body.get('start_time', 0.0)
        duration = body.get('duration', 0.0)
        depth = body.get('depth', 0.0)
        description = body.get('description')
        adapter = self._fI1O4B3(_fOOO4B4)
        try:
            npt_category = NPTCategory[category.upper()]
        except KeyError:
            npt_category = NPTCategory.OTHER
        adapter._fl0l4c6(category=npt_category, start_time=start_time, duration=duration, depth=depth, description=description)
        return APIResponse(success=True, data={'message': 'NPT event added'})

    async def _fIII4c7(self, _fOlO4B6: Dict) -> APIResponse:
        params = _fOlO4B6.get('params', {})
        _fOOO4B4 = params.get('adapter_id')
        adapter = self._fI1O4B3(_fOOO4B4)
        breakdown = adapter._fIII4c7()
        return APIResponse(success=True, data={'breakdown': breakdown})

    async def _f0lO4c8(self, _fOlO4B6: Dict) -> APIResponse:
        params = _fOlO4B6.get('params', {})
        _fOOO4B4 = params.get('adapter_id')
        adapter = self._fI1O4B3(_fOOO4B4)
        return APIResponse(success=True, data=adapter.get_pipe_tally_summary())

    async def _f00O4c9(self, _fOlO4B6: Dict) -> APIResponse:
        body = _fOlO4B6.get('body', {})
        _fOOO4B4 = body.get('adapter_id')
        depth = body.get('depth', 0.0)
        adapter = self._fI1O4B3(_fOOO4B4)
        tally_depth = adapter._pipe_tally.total_length
        difference = abs(depth - tally_depth)
        valid = difference < 1.0
        return APIResponse(success=True, data={'reported_depth': depth, 'tally_depth': tally_depth, 'difference': difference, 'valid': valid})

    async def _fO014cA(self, _fOlO4B6: Dict) -> APIResponse:
        params = _fOlO4B6.get('params', {})
        _fOOO4B4 = params.get('adapter_id')
        adapter = self._fI1O4B3(_fOOO4B4)
        return APIResponse(success=True, data=adapter._fO014cA())

    async def _f1O04cB(self, _fOlO4B6: Dict) -> APIResponse:
        body = _fOlO4B6.get('body', {})
        _fOOO4B4 = body.get('adapter_id')
        measurement = body.get('measurement', 'depth')
        value = body.get('value', 0.0)
        adapter = self._fI1O4B3(_fOOO4B4)
        uncertain = adapter._f1O04cB(measurement, value)
        return APIResponse(success=True, data={'value': uncertain.value, 'uncertainty': uncertain.uncertainty, 'relative_uncertainty': uncertain.relative_uncertainty, 'confidence_interval_95': uncertain.confidence_interval_95})

    async def _fl1l4cc(self, _fOlO4B6: Dict) -> APIResponse:
        params = _fOlO4B6.get('params', {})
        _fOOO4B4 = params.get('adapter_id')
        adapter = self._fI1O4B3(_fOOO4B4)
        return APIResponse(success=True, data=adapter._fl1l4cc())

    async def _fOOI4cd(self, _fOlO4B6: Dict) -> APIResponse:
        params = _fOlO4B6.get('params', {})
        _fOOO4B4 = params.get('adapter_id')
        adapter = self._fI1O4B3(_fOOO4B4)
        return APIResponse(success=True, data={'drilling_efficiency': adapter.get_drilling_efficiency(), 'npt_breakdown': adapter._fIII4c7()})

    async def _flOO4cE(self, _fOlO4B6: Dict) -> APIResponse:
        return APIResponse(success=True, data={'vram_presets': list(VRAM_PRESETS.keys()), 'inference_presets': list(INFERENCE_PRESETS.keys())})

    async def _fIl14cf(self, _fOlO4B6: Dict) -> APIResponse:
        body = _fOlO4B6.get('body', {})
        model_params = body.get('model_params', 100000000)
        vram_preset = body.get('vram_preset', 'rtx_4070_12gb')
        target_batch_size = body.get('target_batch_size', 32)
        recommendation = optimize_for_vram(model_params=model_params, vram_preset=vram_preset, target_batch_size=target_batch_size)
        result = {'recommended_quantization': recommendation['recommended_quantization'], 'fits_in_vram': recommendation['fits_in_vram'], 'max_batch_size': recommendation['max_batch_size'], 'recommended_batch_size': recommendation['recommended_batch_size'], 'gradient_checkpointing': recommendation['gradient_checkpointing'], 'vram_preset': recommendation['vram_preset'], 'memory_breakdown': recommendation['memory_breakdown']}
        return APIResponse(success=True, data=result)
drilling_handler = _cI104Bl()

def _f1104dO() -> List[Route]:
    return drilling_handler._fOO04B2()