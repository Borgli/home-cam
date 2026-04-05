import { MousePointer2, Square, Box, Minus, Trash2 } from 'lucide-react';
import { useStore } from '../../stores/store';

const tools = [
  { id: 'pointer', icon: MousePointer2, label: 'Select', color: 'text-white' },
  { id: 'privacy', icon: Square, label: 'Privacy Zone', color: 'text-neon-red' },
  { id: 'counting', icon: Box, label: 'Counting Zone', color: 'text-neon-green' },
  { id: 'crossing_line', icon: Minus, label: 'Crossing Line', color: 'text-neon-cyan' },
];

export default function DrawingToolbar() {
  const activeTool = useStore(s => s.activeTool);
  const setActiveTool = useStore(s => s.setActiveTool);
  const zones = useStore(s => s.zones);
  const setZones = useStore(s => s.setZones);

  return (
    <div className="flex items-center gap-1 bg-surface rounded-lg border border-border p-1">
      {tools.map(({ id, icon: Icon, label, color }) => (
        <button
          key={id}
          onClick={() => setActiveTool(id)}
          title={label}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all duration-200 ${
            activeTool === id
              ? 'bg-white/10 text-white shadow-inner'
              : 'text-gray-500 hover:text-gray-300 hover:bg-white/5'
          }`}
        >
          <Icon className="w-4 h-4" />
          <span className="hidden sm:inline">{label}</span>
        </button>
      ))}

      {zones.length > 0 && (
        <>
          <div className="w-px h-6 bg-border mx-1" />
          <button
            onClick={() => {
              if (confirm('Clear all zones and lines?')) setZones([]);
            }}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs text-gray-500 hover:text-neon-red hover:bg-neon-red/5 transition-all"
          >
            <Trash2 className="w-4 h-4" />
            <span className="hidden sm:inline">Clear All ({zones.length})</span>
          </button>
        </>
      )}
    </div>
  );
}
