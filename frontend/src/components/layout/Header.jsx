import { useStore } from '../../stores/store';
import { api } from '../../api/client';
import { Eye, EyeOff, Crosshair, Zap, Gauge } from 'lucide-react';

function StatusPill({ active, label, icon: Icon, onClick, color = 'green' }) {
  const colors = {
    green: active ? 'bg-neon-green/15 text-neon-green border-neon-green/30' : 'bg-white/5 text-gray-500 border-border',
    cyan: active ? 'bg-neon-cyan/15 text-neon-cyan border-neon-cyan/30' : 'bg-white/5 text-gray-500 border-border',
    magenta: active ? 'bg-neon-magenta/15 text-neon-magenta border-neon-magenta/30' : 'bg-white/5 text-gray-500 border-border',
    amber: active ? 'bg-neon-amber/15 text-neon-amber border-neon-amber/30' : 'bg-white/5 text-gray-500 border-border',
  };

  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full border text-xs font-medium font-mono transition-all duration-200 hover:scale-105 ${colors[color]}`}
    >
      <Icon className="w-3.5 h-3.5" />
      {label}
    </button>
  );
}

export default function Header() {
  const config = useStore(s => s.config);
  const batchStats = useStore(s => s.batchStats);
  const setConfig = useStore(s => s.setConfig);

  const toggle = async (fn, key) => {
    try {
      const data = await fn();
      setConfig(data);
    } catch { /* backend unavailable */ }
  };

  return (
    <header className="h-14 bg-surface border-b border-border flex items-center justify-between px-4 z-40">
      <div className="flex items-center gap-2">
        <StatusPill
          active={config.detection_enabled}
          label={config.detection_enabled ? 'Detection ON' : 'Detection OFF'}
          icon={config.detection_enabled ? Eye : EyeOff}
          onClick={() => toggle(api.toggleDetection, 'detection_enabled')}
          color="green"
        />
        <StatusPill
          active={config.tracking_enabled}
          label="Tracking"
          icon={Crosshair}
          onClick={() => toggle(api.toggleTracking, 'tracking_enabled')}
          color="magenta"
        />
        <StatusPill
          active={config.batched_mode}
          label={config.batched_mode ? 'Batched' : 'Sequential'}
          icon={Zap}
          onClick={() => toggle(api.toggleBatchedMode, 'batched_mode')}
          color="amber"
        />
        <StatusPill
          active={config.auto_fps}
          label={config.auto_fps ? 'Auto FPS' : `${config.target_fps} FPS`}
          icon={Gauge}
          onClick={() => toggle(api.toggleAutoFps, 'auto_fps')}
          color="cyan"
        />
      </div>

      {/* Batch stats */}
      {batchStats && (
        <div className="flex items-center gap-4 text-xs font-mono text-gray-500">
          <span>Batch: <span className="text-neon-amber">{batchStats.avg_batch_time}ms</span></span>
          <span>Inference: <span className="text-neon-cyan">{batchStats.avg_inference_time}ms</span></span>
          <span>Efficiency: <span className="text-neon-green">{batchStats.avg_efficiency}x</span></span>
          {batchStats.gpu_memory !== '0' && (
            <span>GPU: <span className="text-neon-magenta">{batchStats.gpu_memory}MB</span></span>
          )}
        </div>
      )}
    </header>
  );
}
