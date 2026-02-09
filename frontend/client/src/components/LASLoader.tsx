import { useState, useCallback, useMemo } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, AlertCircle, CheckCircle, Search, ChevronDown, ChevronRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { getApiUrl } from '@/lib/api';
import { CATEGORY_COLORS, CATEGORY_LABELS, type CurveCategory } from '@/lib/lasMnemonics';
import type { LASFileState, LASCurve, DrillingRecord } from '@/types';

type DataMode = 'replace' | 'overlay';

interface LASLoaderProps {
  onDataLoaded: (records: DrillingRecord[], mode: DataMode) => void;
  onCurveDataLoaded?: (fileState: LASFileState, selectedCurves: string[]) => void;
}

export function LASLoader({ onDataLoaded, onCurveDataLoaded }: LASLoaderProps) {
  const [fileState, setFileState] = useState<LASFileState | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [selectedCurves, setSelectedCurves] = useState<Set<string>>(new Set());
  const [dataMode, setDataMode] = useState<DataMode>('replace');
  const [searchFilter, setSearchFilter] = useState('');
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set());
  const [depthRange, setDepthRange] = useState<[number, number]>([0, 0]);
  const [mappingInProgress, setMappingInProgress] = useState(false);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setError(null);
    setSuccess(null);
    setLoading(true);
    setFileState(null);
    setSelectedCurves(new Set());

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(getApiUrl('/api/v1/las/upload'), {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Upload failed');
      }

      const data = await res.json();
      const meta = data.metadata;
      const state: LASFileState = {
        file_id: data.file_id,
        well_name: meta.well_name,
        company: meta.company,
        index_type: meta.index_type,
        index_unit: meta.index_unit,
        index_min: meta.index_min,
        index_max: meta.index_max,
        num_rows: meta.num_rows,
        num_curves: meta.num_curves,
        curves: meta.curves,
      };

      setFileState(state);
      setDepthRange([meta.index_min, meta.index_max]);

      // Auto-select curves that have auto_map_field
      const autoSelected = new Set<string>();
      for (const c of meta.curves) {
        if (c.auto_map_field && c.is_numeric) autoSelected.add(c.mnemonic);
      }
      setSelectedCurves(autoSelected);

      // Auto-expand categories that have auto-mapped curves
      const cats = new Set<string>();
      for (const c of meta.curves) {
        if (autoSelected.has(c.mnemonic)) cats.add(c.category);
      }
      setExpandedCategories(cats);

      setSuccess(`Loaded ${meta.num_curves} curves from "${meta.well_name}"`);
    } catch (e: any) {
      setError(e.message || 'Failed to upload LAS file');
    } finally {
      setLoading(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/octet-stream': ['.las'] },
    multiple: false,
  });

  // Group curves by category
  const groupedCurves = useMemo(() => {
    if (!fileState) return new Map<CurveCategory, LASCurve[]>();
    const groups = new Map<CurveCategory, LASCurve[]>();
    const filtered = fileState.curves.filter(c => {
      if (!searchFilter) return true;
      const q = searchFilter.toLowerCase();
      return c.mnemonic.toLowerCase().includes(q)
        || c.description.toLowerCase().includes(q)
        || c.category.toLowerCase().includes(q)
        || (c.auto_map_field && c.auto_map_field.toLowerCase().includes(q));
    });
    for (const c of filtered) {
      const cat = c.category as CurveCategory;
      if (!groups.has(cat)) groups.set(cat, []);
      groups.get(cat)!.push(c);
    }
    return groups;
  }, [fileState, searchFilter]);

  const toggleCurve = (mnemonic: string) => {
    setSelectedCurves(prev => {
      const next = new Set(prev);
      if (next.has(mnemonic)) next.delete(mnemonic);
      else next.add(mnemonic);
      return next;
    });
  };

  const toggleCategory = (cat: string) => {
    setExpandedCategories(prev => {
      const next = new Set(prev);
      if (next.has(cat)) next.delete(cat);
      else next.add(cat);
      return next;
    });
  };

  const selectAllInCategory = (cat: CurveCategory) => {
    const curves = groupedCurves.get(cat);
    if (!curves) return;
    setSelectedCurves(prev => {
      const next = new Set(prev);
      const allSelected = curves.every(c => next.has(c.mnemonic));
      for (const c of curves) {
        if (allSelected) next.delete(c.mnemonic);
        else next.add(c.mnemonic);
      }
      return next;
    });
  };

  const handleMapToDrilling = async () => {
    if (!fileState || selectedCurves.size === 0) return;
    setMappingInProgress(true);
    setError(null);

    // Build curve_mapping from auto_map_field
    const curveMapping: Record<string, string> = {};
    for (const c of fileState.curves) {
      if (selectedCurves.has(c.mnemonic) && c.auto_map_field) {
        curveMapping[c.auto_map_field] = c.mnemonic;
      }
    }

    if (Object.keys(curveMapping).length === 0) {
      setError('No auto-mappable curves selected. Select curves with auto-map targets (WOB, ROP, RPM, etc.).');
      setMappingInProgress(false);
      return;
    }

    try {
      const res = await fetch(getApiUrl(`/api/v1/las/${fileState.file_id}/map-to-drilling`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          curve_mapping: curveMapping,
          start_index: depthRange[0],
          end_index: depthRange[1],
          max_points: 5000,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Mapping failed');
      }

      const data = await res.json();
      onDataLoaded(data.records as DrillingRecord[], dataMode);
      setSuccess(`Mapped ${data.count} records to drilling format. ${data.regime ? `Regime: ${data.regime.regime} (${(data.regime.confidence * 100).toFixed(0)}%)` : ''}`);
    } catch (e: any) {
      setError(e.message || 'Failed to map to drilling records');
    } finally {
      setMappingInProgress(false);
    }
  };

  const handleLoadCurves = () => {
    if (fileState && selectedCurves.size > 0 && onCurveDataLoaded) {
      onCurveDataLoaded(fileState, Array.from(selectedCurves));
    }
  };

  return (
    <div className="space-y-4 max-h-[70vh] overflow-y-auto">
      {/* Drop zone */}
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-md p-6 text-center cursor-pointer transition-colors
          ${isDragActive ? 'border-primary bg-primary/10' : 'border-muted-foreground/25 hover:border-primary/50'}
        `}
      >
        <input {...getInputProps()} />
        <FileText className="w-8 h-8 mx-auto mb-2 text-muted-foreground" />
        {loading ? (
          <div className="flex items-center justify-center gap-2">
            <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />
            <span className="text-sm text-primary">Parsing LAS file...</span>
          </div>
        ) : isDragActive ? (
          <p className="text-sm text-primary font-medium">Drop the LAS file here...</p>
        ) : (
          <div className="space-y-1">
            <p className="text-sm text-muted-foreground">Drag & drop .LAS files</p>
            <p className="text-xs text-muted-foreground/50">Supports LAS 2.0 and 3.0 formats</p>
          </div>
        )}
      </div>

      {/* Error / Success alerts */}
      {error && (
        <Alert variant="destructive" className="py-2">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
      {success && (
        <Alert className="py-2 border-green-500/50 text-green-500">
          <CheckCircle className="h-4 w-4" />
          <AlertTitle>Success</AlertTitle>
          <AlertDescription>{success}</AlertDescription>
        </Alert>
      )}

      {/* Well metadata header */}
      {fileState && (
        <>
          <div className="bg-card/50 border border-border rounded-md p-3 space-y-2">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-mono text-sm font-bold">{fileState.well_name || 'Unknown Well'}</h3>
                {fileState.company && (
                  <p className="text-xs text-muted-foreground">{fileState.company}</p>
                )}
              </div>
              <div className="flex items-center gap-2">
                <span className={`text-xs px-2 py-0.5 rounded-full font-mono ${
                  fileState.index_type === 'DEPTH' ? 'bg-blue-500/20 text-blue-400' : 'bg-amber-500/20 text-amber-400'
                }`}>
                  {fileState.index_type}
                </span>
                <span className="text-xs text-muted-foreground font-mono">
                  {fileState.num_curves} curves
                </span>
              </div>
            </div>
            <div className="flex gap-4 text-xs text-muted-foreground font-mono">
              <span>Range: {fileState.index_min.toFixed(1)} - {fileState.index_max.toFixed(1)} {fileState.index_unit}</span>
              <span>Rows: {fileState.num_rows.toLocaleString()}</span>
            </div>
          </div>

          {/* Data mode toggle */}
          <div className="flex items-center gap-3">
            <span className="text-xs font-mono text-muted-foreground">MODE:</span>
            <label className="flex items-center gap-1.5 cursor-pointer">
              <input
                type="radio"
                name="dataMode"
                checked={dataMode === 'replace'}
                onChange={() => setDataMode('replace')}
                className="accent-primary"
              />
              <span className="text-xs font-mono">Replace data</span>
            </label>
            <label className="flex items-center gap-1.5 cursor-pointer">
              <input
                type="radio"
                name="dataMode"
                checked={dataMode === 'overlay'}
                onChange={() => setDataMode('overlay')}
                className="accent-primary"
              />
              <span className="text-xs font-mono">Overlay</span>
            </label>
          </div>

          {/* Depth/time window */}
          <div className="space-y-1">
            <span className="text-xs font-mono text-muted-foreground">WINDOW:</span>
            <div className="flex items-center gap-2">
              <input
                type="number"
                value={depthRange[0]}
                onChange={e => setDepthRange([Number(e.target.value), depthRange[1]])}
                className="w-24 bg-background border border-border rounded px-2 py-1 text-xs font-mono"
              />
              <span className="text-xs text-muted-foreground">to</span>
              <input
                type="number"
                value={depthRange[1]}
                onChange={e => setDepthRange([depthRange[0], Number(e.target.value)])}
                className="w-24 bg-background border border-border rounded px-2 py-1 text-xs font-mono"
              />
              <span className="text-xs text-muted-foreground">{fileState.index_unit}</span>
            </div>
          </div>

          {/* Search filter */}
          <div className="relative">
            <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground" />
            <input
              type="text"
              placeholder="Filter curves..."
              value={searchFilter}
              onChange={e => setSearchFilter(e.target.value)}
              className="w-full bg-background border border-border rounded px-7 py-1.5 text-xs font-mono"
            />
          </div>

          {/* Curve catalog */}
          <div className="border border-border rounded-md divide-y divide-border max-h-[300px] overflow-y-auto">
            {Array.from(groupedCurves.entries()).map(([cat, curves]) => {
              const isExpanded = expandedCategories.has(cat);
              const selectedCount = curves.filter(c => selectedCurves.has(c.mnemonic)).length;
              return (
                <div key={cat}>
                  <button
                    onClick={() => toggleCategory(cat)}
                    className="w-full flex items-center justify-between px-3 py-2 hover:bg-accent/50 transition-colors"
                  >
                    <div className="flex items-center gap-2">
                      {isExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                      <span
                        className="w-2 h-2 rounded-full"
                        style={{ backgroundColor: CATEGORY_COLORS[cat] }}
                      />
                      <span className="text-xs font-mono font-medium">{CATEGORY_LABELS[cat] || cat}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] text-muted-foreground">{selectedCount}/{curves.length}</span>
                      <button
                        onClick={e => { e.stopPropagation(); selectAllInCategory(cat); }}
                        className="text-[10px] text-primary hover:underline"
                      >
                        {selectedCount === curves.length ? 'none' : 'all'}
                      </button>
                    </div>
                  </button>
                  {isExpanded && (
                    <div className="bg-card/30 divide-y divide-border/30">
                      {curves.map(c => (
                        <label
                          key={c.mnemonic}
                          className="flex items-center gap-3 px-6 py-1.5 hover:bg-accent/30 cursor-pointer"
                        >
                          <input
                            type="checkbox"
                            checked={selectedCurves.has(c.mnemonic)}
                            onChange={() => toggleCurve(c.mnemonic)}
                            className="accent-primary w-3.5 h-3.5"
                          />
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                              <span className="text-xs font-mono font-medium truncate">{c.mnemonic}</span>
                              {c.unit && (
                                <span className="text-[10px] text-muted-foreground">[{c.unit}]</span>
                              )}
                              {c.auto_map_field && (
                                <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/20 text-primary">
                                  {c.auto_map_field}
                                </span>
                              )}
                            </div>
                            {c.description && (
                              <p className="text-[10px] text-muted-foreground truncate">{c.description}</p>
                            )}
                          </div>
                          <div className="flex items-center gap-2 text-[10px] text-muted-foreground font-mono shrink-0">
                            {c.is_numeric && c.min_val !== null && (
                              <span>{c.min_val.toFixed(1)}-{c.max_val?.toFixed(1)}</span>
                            )}
                            <span className={c.null_pct > 50 ? 'text-red-400' : c.null_pct > 10 ? 'text-yellow-400' : ''}>
                              {c.null_pct.toFixed(0)}% null
                            </span>
                          </div>
                        </label>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Action buttons */}
          <div className="flex items-center gap-2 pt-2">
            <Button
              size="sm"
              variant="outline"
              className="font-mono text-xs"
              disabled={selectedCurves.size === 0 || !onCurveDataLoaded}
              onClick={handleLoadCurves}
            >
              <Upload className="w-3 h-3 mr-1.5" />
              LOAD CURVES ({selectedCurves.size})
            </Button>
            <Button
              size="sm"
              className="font-mono text-xs"
              disabled={selectedCurves.size === 0 || mappingInProgress}
              onClick={handleMapToDrilling}
            >
              {mappingInProgress ? (
                <div className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin mr-1.5" />
              ) : null}
              MAP TO DRILLING
            </Button>
          </div>
          <p className="text-[10px] text-muted-foreground">
            {selectedCurves.size} curve{selectedCurves.size !== 1 ? 's' : ''} selected.
            {' '}Mapped fields: {fileState.curves.filter(c => selectedCurves.has(c.mnemonic) && c.auto_map_field).map(c => c.auto_map_field).join(', ') || 'none'}
          </p>
        </>
      )}
    </div>
  );
}
