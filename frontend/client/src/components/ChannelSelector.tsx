import { useState } from 'react';
import { cn } from '@/lib/utils';

// -- Channel taxonomy -------------------------------------------------------

export interface ChannelDef {
  id: string;
  label: string;
  category: string;
}

const ALL_CHANNELS: ChannelDef[] = [
  // Mechanical
  { id: 'WOB', label: 'Weight on Bit', category: 'mechanical' },
  { id: 'TRQ', label: 'Torque', category: 'mechanical' },
  { id: 'RPM', label: 'Rotary Speed', category: 'mechanical' },
  { id: 'HKLD', label: 'Hookload', category: 'mechanical' },
  // Hydraulic
  { id: 'SPP', label: 'Standpipe Pressure', category: 'hydraulic' },
  { id: 'FLOW', label: 'Flow Rate', category: 'hydraulic' },
  { id: 'ECD', label: 'Equiv Circ Density', category: 'hydraulic' },
  { id: 'PUMP', label: 'Pump Strokes', category: 'hydraulic' },
  // Formation
  { id: 'GR', label: 'Gamma Ray', category: 'formation' },
  { id: 'RES', label: 'Resistivity', category: 'formation' },
  // Directional
  { id: 'INC', label: 'Inclination', category: 'directional' },
  { id: 'AZI', label: 'Azimuth', category: 'directional' },
  { id: 'DLS', label: 'Dogleg Severity', category: 'directional' },
  // Vibration
  { id: 'VIB', label: 'Lateral Vibration', category: 'vibration' },
  { id: 'SS', label: 'Stick-Slip Index', category: 'vibration' },
  // Performance
  { id: 'ROP', label: 'Rate of Penetration', category: 'performance' },
  { id: 'MSE', label: 'Mech Specific Energy', category: 'performance' },
  { id: 'DEPTH', label: 'Measured Depth', category: 'performance' },
];

const CATEGORY_COLORS: Record<string, string> = {
  mechanical: '#f97316',
  hydraulic: '#3b82f6',
  formation: '#a855f7',
  directional: '#06b6d4',
  vibration: '#ec4899',
  performance: '#22c55e',
};

// -- Stakeholder presets ----------------------------------------------------

interface Preset {
  id: string;
  label: string;
  channels: string[];
}

const PRESETS: Preset[] = [
  { id: 'company_man', label: 'Co Man', channels: ['ROP', 'WOB', 'RPM', 'TRQ', 'DEPTH', 'MSE'] },
  { id: 'dd', label: 'DD', channels: ['INC', 'AZI', 'DLS', 'WOB', 'RPM', 'TRQ', 'ROP', 'SPP'] },
  { id: 'mwd', label: 'MWD', channels: ['GR', 'RES', 'INC', 'AZI', 'VIB', 'SS', 'FLOW'] },
  { id: 'gen_super', label: 'Gen Super', channels: ['ROP', 'DEPTH', 'MSE', 'WOB', 'SPP', 'ECD'] },
  {
    id: 'office_engineer',
    label: 'Office',
    channels: ['ROP', 'MSE', 'DEPTH', 'WOB', 'TRQ', 'RPM', 'SPP', 'FLOW', 'GR'],
  },
  { id: 'all', label: 'All', channels: ALL_CHANNELS.map((c) => c.id) },
];

// -- Component --------------------------------------------------------------

interface ChannelSelectorProps {
  selected: string[];
  onChange: (channels: string[]) => void;
  className?: string;
}

export function ChannelSelector({ selected, onChange, className }: ChannelSelectorProps) {
  const [open, setOpen] = useState(false);

  const activePreset = PRESETS.find(
    (p) =>
      p.channels.length === selected.length &&
      p.channels.every((c) => selected.includes(c)),
  );

  const toggle = (id: string) => {
    if (selected.includes(id)) {
      onChange(selected.filter((c) => c !== id));
    } else {
      onChange([...selected, id]);
    }
  };

  const grouped = ALL_CHANNELS.reduce<Record<string, ChannelDef[]>>((acc, ch) => {
    (acc[ch.category] ??= []).push(ch);
    return acc;
  }, {});

  return (
    <div className={cn('relative', className)}>
      {/* Trigger */}
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-2 px-2 py-0.5 text-[10px] font-mono border border-border hover:border-primary/40 transition-colors bg-card/60"
      >
        <span className="text-muted-foreground">CHANNELS:</span>
        <span className="text-primary">{activePreset?.label ?? 'Custom'}</span>
        <span className="text-muted-foreground">({selected.length})</span>
        <span className="text-muted-foreground text-[8px]">{open ? '\u25B2' : '\u25BC'}</span>
      </button>

      {/* Popover */}
      {open && (
        <div className="absolute top-full left-0 mt-1 z-50 w-72 bg-card border border-border shadow-lg font-mono text-[10px]">
          {/* Presets */}
          <div className="flex flex-wrap gap-1 px-2 py-1.5 border-b border-border/50">
            {PRESETS.map((p) => (
              <button
                key={p.id}
                onClick={() => {
                  onChange(p.channels);
                }}
                className={cn(
                  'px-1.5 py-0.5 border transition-colors',
                  activePreset?.id === p.id
                    ? 'border-primary bg-primary/20 text-primary'
                    : 'border-border text-muted-foreground hover:text-foreground',
                )}
              >
                {p.label}
              </button>
            ))}
          </div>

          {/* Grouped checkboxes */}
          <div className="max-h-64 overflow-y-auto px-2 py-1.5 space-y-2">
            {Object.entries(grouped).map(([cat, channels]) => (
              <div key={cat}>
                <div
                  className="text-[9px] uppercase tracking-wider mb-0.5 font-bold"
                  style={{ color: CATEGORY_COLORS[cat] ?? '#888' }}
                >
                  {cat}
                </div>
                {channels.map((ch) => (
                  <label
                    key={ch.id}
                    className="flex items-center gap-1.5 py-0.5 cursor-pointer hover:bg-primary/5"
                  >
                    <input
                      type="checkbox"
                      checked={selected.includes(ch.id)}
                      onChange={() => toggle(ch.id)}
                      className="w-3 h-3 accent-primary"
                    />
                    <span className="text-foreground">{ch.id}</span>
                    <span className="text-muted-foreground ml-auto">{ch.label}</span>
                  </label>
                ))}
              </div>
            ))}
          </div>

          {/* Footer */}
          <div className="flex justify-between items-center px-2 py-1 border-t border-border/50">
            <span className="text-muted-foreground">{selected.length} selected</span>
            <button
              onClick={() => setOpen(false)}
              className="text-primary hover:underline"
            >
              Done
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
