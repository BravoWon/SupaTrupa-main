"""LAS file parser with windowed reading, mnemonic aliasing, and DrillingRecord mapping."""

from __future__ import annotations

import hashlib
import math
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import lasio
except ImportError:
    lasio = None  # type: ignore


class CurveCategory(str, Enum):
    DEPTH = "DEPTH"
    ROP = "ROP"
    WEIGHT = "WEIGHT"
    TORQUE = "TORQUE"
    ROTARY = "ROTARY"
    PRESSURE = "PRESSURE"
    FLOW = "FLOW"
    MUD = "MUD"
    VIBRATION = "VIBRATION"
    GAMMA = "GAMMA"
    DIRECTIONAL = "DIRECTIONAL"
    TEMPERATURE = "TEMPERATURE"
    AUTODRILLER = "AUTODRILLER"
    TANK_VOLUMES = "TANK_VOLUMES"
    PUMP = "PUMP"
    GAS = "GAS"
    OTHER = "OTHER"


# Maps canonical DrillingRecord field name → list of known LAS mnemonics
MNEMONIC_ALIASES: Dict[str, List[str]] = {
    "depth": ["DEPT", "NOMD", "Hole.ft", "DEPTH", "MD", "DMEA", "DBTM"],
    "wob": ["WOB", "WOBX", "SWOB", "Bit.klb", "WOB_AVG", "WOBR"],
    "rpm": ["RPM", "Rota.RPM", "Top.RPM", "TRPM", "SRPM", "RPM_AVG", "ROTARY_RPM"],
    "rop": ["ROP", "OBR", "ROP_AVG", "ROPA", "ROPX", "APTS", "ROPI", "ROP5"],
    "torque": ["TQA", "TOR", "Rota.A", "Top.ft-lbf", "TORQUE", "TQON", "TQOFF", "TRQ"],
    "spp": ["SPPA", "SPP", "Pump.psi", "SPRESS", "PPRS", "STANDPIPE"],
    "hookload": ["HKLD", "HKL", "HL", "Hook.klb", "HOOKLOAD", "HKLM"],
}

# Reverse lookup: mnemonic → (drilling_field, category)
_MNEMONIC_TO_FIELD: Dict[str, str] = {}
for _field, _aliases in MNEMONIC_ALIASES.items():
    for _alias in _aliases:
        _MNEMONIC_TO_FIELD[_alias.upper()] = _field

# Mnemonic → category classification
_CATEGORY_PATTERNS: Dict[CurveCategory, List[str]] = {
    CurveCategory.DEPTH: ["DEPT", "NOMD", "DEPTH", "MD", "DMEA", "DBTM", "TVD", "HOLE"],
    CurveCategory.ROP: ["ROP", "OBR", "APTS", "ROPA", "ROPX", "ROP5"],
    CurveCategory.WEIGHT: ["WOB", "WOBX", "SWOB", "HKLD", "HKL", "HL", "HOOKLOAD"],
    CurveCategory.TORQUE: ["TQA", "TOR", "TORQUE", "TQON", "TQOFF", "TRQ"],
    CurveCategory.ROTARY: ["RPM", "TRPM", "SRPM", "ROTARY"],
    CurveCategory.PRESSURE: ["SPP", "PPRS", "SPRESS", "STANDPIPE", "ECD", "PRESS", "BHP"],
    CurveCategory.FLOW: ["FLOW", "FLOWIN", "FLOWOUT", "GPM", "MFOP", "MFIP", "TFLO"],
    CurveCategory.MUD: [
        "MUD", "MWIN", "MWOUT", "MW", "DENSITY", "VISC",
        "RHEO", "PV", "YP", "GELS", "COND",
    ],
    CurveCategory.VIBRATION: ["VIB", "AXIAL", "LATERAL", "STICK", "TORS"],
    CurveCategory.GAMMA: ["GR", "GAMMA", "GRD", "GAPI", "SGR"],
    CurveCategory.DIRECTIONAL: [
        "INC", "INCL", "AZI", "AZIM", "DLS", "TFO",
        "GTOT", "BTOT", "MAG", "GRAV", "DIPTOT",
    ],
    CurveCategory.TEMPERATURE: ["TEMP", "BHT", "MTEMP", "STEMP", "ANNTEMP"],
    CurveCategory.AUTODRILLER: ["ADR", "ADSB", "ADWOB", "ADTOR", "AUTO"],
    CurveCategory.TANK_VOLUMES: ["TANK", "PIT", "TVol", "ACTIVE", "RESERVE", "TRIP"],
    CurveCategory.PUMP: ["PUMP", "PSI", "SPM", "STROKES", "LINER"],
    CurveCategory.GAS: ["GAS", "C1", "C2", "C3", "C4", "C5", "H2S", "CO2", "TGS"],
}


def _classify_mnemonic(mnemonic: str) -> CurveCategory:
    upper = mnemonic.upper().replace(".", "_")
    for cat, patterns in _CATEGORY_PATTERNS.items():
        for pat in patterns:
            if pat in upper:
                return cat
    return CurveCategory.OTHER


def _auto_map_field(mnemonic: str) -> Optional[str]:
    return _MNEMONIC_TO_FIELD.get(mnemonic.upper(), None)


@dataclass
class CurveInfo:
    mnemonic: str
    unit: str
    description: str
    category: CurveCategory
    is_numeric: bool
    null_pct: float
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    auto_map_field: Optional[str] = None


@dataclass
class LASFileMetadata:
    file_id: str
    well_name: str
    company: str
    index_type: str  # "DEPTH" or "TIME"
    index_unit: str
    index_min: float
    index_max: float
    num_rows: int
    curves: List[CurveInfo] = field(default_factory=list)


# --- In-memory LRU cache for parsed LAS files ---
_LAS_CACHE: OrderedDict[str, Any] = OrderedDict()
_MAX_CACHE = 3


def _cache_put(file_id: str, obj: Any) -> None:
    if file_id in _LAS_CACHE:
        _LAS_CACHE.move_to_end(file_id)
    else:
        _LAS_CACHE[file_id] = obj
        if len(_LAS_CACHE) > _MAX_CACHE:
            _LAS_CACHE.popitem(last=False)


def _cache_get(file_id: str) -> Any:
    if file_id in _LAS_CACHE:
        _LAS_CACHE.move_to_end(file_id)
        return _LAS_CACHE[file_id]
    return None


def _file_id(filepath: str) -> str:
    basename = os.path.basename(filepath)
    stat = os.stat(filepath)
    raw = f"{basename}:{stat.st_size}:{stat.st_mtime}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _is_las3_tab(filepath: str) -> bool:
    """Detect LAS 3.0 TAB-delimited files by scanning first few lines."""
    try:
        with open(filepath, "r", errors="replace") as f:
            for _ in range(20):
                line = f.readline()
                if not line:
                    break
                stripped = line.strip()
                if stripped.startswith("~A") or stripped.startswith("~ASCII"):
                    if "TAB" in stripped.upper() or "DLM" in stripped.upper():
                        return True
    except Exception:
        pass
    return False


def _read_las3_tab_data(filepath: str) -> Optional[pd.DataFrame]:
    """Read LAS 3.0 TAB data section using pandas for speed."""
    header_lines: List[str] = []
    data_start = 0
    col_names: List[str] = []

    with open(filepath, "r", errors="replace") as f:
        in_curve_section = False
        for i, line in enumerate(f):
            stripped = line.strip()
            if stripped.startswith("~C") or stripped.startswith("~Curve"):
                in_curve_section = True
                continue
            if in_curve_section:
                if stripped.startswith("~"):
                    in_curve_section = False
                elif stripped and not stripped.startswith("#"):
                    # Parse curve def: MNEMONIC.UNIT  ...
                    parts = stripped.split(".")
                    if parts:
                        col_names.append(parts[0].strip())

            if stripped.startswith("~A") or stripped.startswith("~ASCII"):
                data_start = i + 1
                break

    if data_start == 0 or not col_names:
        return None

    try:
        df = pd.read_csv(
            filepath,
            sep="\t",
            skiprows=data_start,
            names=col_names if col_names else None,
            header=None if col_names else 0,
            na_values=["-999.25", "-999.2500", "-999", ""],
            comment="#",
            on_bad_lines="skip",
        )
        return df
    except Exception:
        return None


def parse_las_metadata(filepath: str) -> LASFileMetadata:
    """Parse LAS file metadata (header only for speed)."""
    if lasio is None:
        raise ImportError("lasio is not installed")

    fid = _file_id(filepath)

    # Try header-only parse first
    try:
        las = lasio.read(filepath, ignore_data=True)
    except Exception:
        las = lasio.read(filepath)

    def _well_val(mnemonic: str, default: str = "") -> str:
        """Safely extract a well header string value."""
        try:
            item = las.well[mnemonic]
            return str(item.value) if item.value else default
        except (KeyError, IndexError):
            return default

    def _well_float(mnemonic: str, default: float = 0.0) -> float:
        """Safely extract a well header numeric value."""
        try:
            item = las.well[mnemonic]
            return float(item.value)
        except (KeyError, IndexError, TypeError, ValueError):
            return default

    well_name = _well_val("WELL") or os.path.basename(filepath)
    company = _well_val("COMP")

    # Determine index type
    index_min = _well_float("STRT")
    index_max = _well_float("STOP")
    try:
        index_unit = str(las.well["STRT"].unit) if las.well["STRT"].unit else ""
    except (KeyError, IndexError):
        index_unit = ""

    time_indicators = ["S", "SEC", "SECONDS", "MS", "MILLISECONDS", "HR", "MIN"]
    is_time = index_unit.upper() in time_indicators
    index_type = "TIME" if is_time else "DEPTH"

    # Get curve stats from partial data read
    curves_info: List[CurveInfo] = []
    try:
        las_full = lasio.read(filepath)
        data = las_full.data
        num_rows = len(data) if data is not None and len(data) > 0 else 0
    except Exception:
        num_rows = 0
        data = None

    for curve in las.curves:
        mnem = curve.mnemonic
        unit = str(curve.unit) if curve.unit else ""
        desc = str(curve.descr) if curve.descr else ""
        category = _classify_mnemonic(mnem)
        auto_field = _auto_map_field(mnem)

        is_numeric = True
        null_pct = 0.0
        min_val = None
        max_val = None

        if data is not None and num_rows > 0:
            try:
                idx = [c.mnemonic for c in las_full.curves].index(mnem)
                col_data = data[:, idx]
                col_data = col_data.astype(float)
                null_mask = np.isnan(col_data) | (col_data == -999.25)
                null_pct = float(null_mask.sum()) / num_rows * 100
                valid = col_data[~null_mask]
                if len(valid) > 0:
                    min_val = float(np.min(valid))
                    max_val = float(np.max(valid))
            except (ValueError, IndexError, TypeError):
                is_numeric = False
                null_pct = 100.0

        curves_info.append(
            CurveInfo(
                mnemonic=mnem,
                unit=unit,
                description=desc,
                category=category,
                is_numeric=is_numeric,
                null_pct=round(null_pct, 1),
                min_val=min_val,
                max_val=max_val,
                auto_map_field=auto_field,
            )
        )

    return LASFileMetadata(
        file_id=fid,
        well_name=well_name,
        company=company,
        index_type=index_type,
        index_unit=index_unit,
        index_min=index_min,
        index_max=index_max,
        num_rows=num_rows,
        curves=curves_info,
    )


def read_las_window(
    filepath: str,
    curves: List[str],
    start: Optional[float] = None,
    end: Optional[float] = None,
    max_points: int = 5000,
) -> Dict[str, Any]:
    """Read windowed curve data with decimation for large files.

    Returns dict with index_name, index_unit, index_values, curves, total_rows, decimation_factor.
    """
    if lasio is None:
        raise ImportError("lasio is not installed")

    fid = _file_id(filepath)

    # Check LRU cache
    cached = _cache_get(fid)
    if cached is not None:
        las = cached
    else:
        # For large LAS 3.0 TAB files, try pandas first
        if _is_las3_tab(filepath) and os.path.getsize(filepath) > 10_000_000:
            df = _read_las3_tab_data(filepath)
            if df is not None:
                _cache_put(fid, ("df", df))
                return _extract_from_df(df, curves, start, end, max_points)

        las = lasio.read(filepath)
        _cache_put(fid, las)

    if isinstance(las, tuple) and las[0] == "df":
        return _extract_from_df(las[1], curves, start, end, max_points)

    # Extract index
    index_curve = las.curves[0]
    index_name = index_curve.mnemonic
    index_unit = str(index_curve.unit) if index_curve.unit else ""
    index_data = las.data[:, 0].astype(float)

    # Window filter
    mask = np.ones(len(index_data), dtype=bool)
    if start is not None:
        mask &= index_data >= start
    if end is not None:
        mask &= index_data <= end

    total_rows = int(mask.sum())

    # Decimation
    decimation_factor = max(1, total_rows // max_points)
    indices = np.where(mask)[0][::decimation_factor]

    result_index = index_data[indices].tolist()

    available_mnemonics = [c.mnemonic for c in las.curves]
    result_curves: Dict[str, dict] = {}
    for mnem in curves:
        if mnem in available_mnemonics:
            col_idx = available_mnemonics.index(mnem)
            col_data = las.data[indices, col_idx].astype(float)
            # Replace -999.25 null values with None
            values = []
            for v in col_data:
                if np.isnan(v) or abs(v - (-999.25)) < 0.01:
                    values.append(None)
                else:
                    values.append(float(v))
            curve_obj = las.curves[col_idx]
            result_curves[mnem] = {
                "values": values,
                "unit": str(curve_obj.unit) if curve_obj.unit else "",
                "category": _classify_mnemonic(mnem).value,
            }

    return {
        "index_name": index_name,
        "index_unit": index_unit,
        "index_values": result_index,
        "curves": result_curves,
        "total_rows": total_rows,
        "decimation_factor": decimation_factor,
    }


def _extract_from_df(
    df: pd.DataFrame,
    curves: List[str],
    start: Optional[float],
    end: Optional[float],
    max_points: int,
) -> Dict[str, Any]:
    """Extract curve data from a pandas DataFrame (LAS 3.0 TAB path)."""
    cols = list(df.columns)
    if not cols:
        return {
            "index_name": "DEPT",
            "index_unit": "",
            "index_values": [],
            "curves": {},
            "total_rows": 0,
            "decimation_factor": 1,
        }

    index_name = cols[0]
    index_data = pd.to_numeric(df[index_name], errors="coerce")

    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= index_data >= start
    if end is not None:
        mask &= index_data <= end

    total_rows = int(mask.sum())
    decimation_factor = max(1, total_rows // max_points)

    filtered = df.loc[mask].iloc[::decimation_factor]

    result_curves: Dict[str, dict] = {}
    for mnem in curves:
        if mnem in cols:
            col_data = pd.to_numeric(filtered[mnem], errors="coerce")
            values = [None if pd.isna(v) else float(v) for v in col_data]
            result_curves[mnem] = {
                "values": values,
                "unit": "",
                "category": _classify_mnemonic(mnem).value,
            }

    return {
        "index_name": index_name,
        "index_unit": "",
        "index_values": pd.to_numeric(filtered[index_name], errors="coerce").tolist(),
        "curves": result_curves,
        "total_rows": total_rows,
        "decimation_factor": decimation_factor,
    }


def map_to_drilling_records(
    filepath: str,
    curve_mapping: Dict[str, str],
    start: Optional[float] = None,
    end: Optional[float] = None,
    max_points: int = 5000,
) -> List[Dict[str, Any]]:
    """Convert LAS curves to DrillingRecord-compatible dicts.

    curve_mapping: {drilling_field: las_mnemonic} e.g. {"wob": "WOB", "rop": "OBR"}

    Synthesizes missing fields (surprise, activity, mu, sigma_trace) from rolling statistics.
    """
    # Figure out which LAS mnemonics to fetch
    las_mnemonics = list(curve_mapping.values())

    # Also grab the index/depth curve
    raw = read_las_window(filepath, las_mnemonics, start, end, max_points)

    index_vals = raw["index_values"]
    n = len(index_vals)
    if n == 0:
        return []

    # Build arrays for each drilling field
    field_arrays: Dict[str, List[Optional[float]]] = {}
    for drilling_field, las_mnem in curve_mapping.items():
        if las_mnem in raw["curves"]:
            field_arrays[drilling_field] = raw["curves"][las_mnem]["values"]
        else:
            field_arrays[drilling_field] = [None] * n

    # Assemble records
    records: List[Dict[str, Any]] = []
    DRILL_FIELDS = ["wob", "rpm", "rop", "torque", "spp", "hookload"]

    # Pre-compute rolling stats for surprise/activity synthesis
    rop_vals = np.array(
        [v if v is not None else np.nan for v in field_arrays.get("rop", [None] * n)],
        dtype=float,
    )
    torque_vals = np.array(
        [v if v is not None else np.nan for v in field_arrays.get("torque", [None] * n)],
        dtype=float,
    )
    wob_vals = np.array(
        [v if v is not None else np.nan for v in field_arrays.get("wob", [None] * n)],
        dtype=float,
    )

    window = 20

    for i in range(n):
        rec: Dict[str, Any] = {
            "id": i,
            "depth": index_vals[i] if index_vals[i] is not None else 0.0,
        }

        for f in DRILL_FIELDS:
            arr = field_arrays.get(f, [None] * n)
            rec[f] = arr[i] if arr[i] is not None else 0.0

        # Synthesize surprise = z-score of ROP in rolling window
        w_start = max(0, i - window)
        rop_window = rop_vals[w_start : i + 1]
        valid_rop = rop_window[~np.isnan(rop_window)]
        if len(valid_rop) > 1:
            rop_mean = np.mean(valid_rop)
            rop_std = np.std(valid_rop)
            current_rop = rop_vals[i]
            if not np.isnan(current_rop) and rop_std > 0:
                rec["surprise"] = float(abs(current_rop - rop_mean) / rop_std)
            else:
                rec["surprise"] = 0.0
        else:
            rec["surprise"] = 0.0

        # Synthesize activity = magnitude of parameter change rate
        if i > 0:
            changes = []
            for arr_name in [rop_vals, torque_vals, wob_vals]:
                prev = arr_name[i - 1] if not np.isnan(arr_name[i - 1]) else 0.0
                curr = arr_name[i] if not np.isnan(arr_name[i]) else 0.0
                changes.append(abs(curr - prev))
            rec["activity"] = float(np.mean(changes))
        else:
            rec["activity"] = 0.0

        # Activity type based on surprise
        if rec["surprise"] > 2.0:
            rec["activity_type"] = "spike"
        elif rec["surprise"] > 0.5:
            rec["activity_type"] = "subthreshold"
        else:
            rec["activity_type"] = "silent"

        # Confidence = 1.0 - (fraction of null fields among drilling fields)
        null_count = sum(1 for f in DRILL_FIELDS if field_arrays.get(f, [None] * n)[i] is None)
        rec["confidence"] = 1.0 - (null_count / len(DRILL_FIELDS))

        # mu = [normalized_rop, normalized_torque, normalized_wob]
        rop_max = np.nanmax(rop_vals) if np.any(~np.isnan(rop_vals)) else 1.0
        torque_max = np.nanmax(torque_vals) if np.any(~np.isnan(torque_vals)) else 1.0
        wob_max = np.nanmax(wob_vals) if np.any(~np.isnan(wob_vals)) else 1.0

        r = rop_vals[i] if not np.isnan(rop_vals[i]) else 0.0
        t = torque_vals[i] if not np.isnan(torque_vals[i]) else 0.0
        w = wob_vals[i] if not np.isnan(wob_vals[i]) else 0.0

        rec["mu"] = [
            float(r / rop_max) if rop_max > 0 else 0.0,
            float(t / torque_max) if torque_max > 0 else 0.0,
            float(w / wob_max) if wob_max > 0 else 0.0,
        ]

        # sigma_trace = rolling std of mu norm
        mu_norms = []
        for j in range(w_start, i + 1):
            rj = rop_vals[j] if not np.isnan(rop_vals[j]) else 0.0
            tj = torque_vals[j] if not np.isnan(torque_vals[j]) else 0.0
            wj = wob_vals[j] if not np.isnan(wob_vals[j]) else 0.0
            mu_norm = math.sqrt(
                (rj / rop_max if rop_max > 0 else 0.0) ** 2
                + (tj / torque_max if torque_max > 0 else 0.0) ** 2
                + (wj / wob_max if wob_max > 0 else 0.0) ** 2
            )
            mu_norms.append(mu_norm)
        rec["sigma_trace"] = float(np.std(mu_norms)) if len(mu_norms) > 1 else 0.0

        records.append(rec)

    return records
