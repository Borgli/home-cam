import { useState } from 'react';
import { useStore } from '../../stores/store';
import { api } from '../../api/client';

export default function FPSControl() {
  const config = useStore(s => s.config);
  const setConfig = useStore(s => s.setConfig);
  const [localFps, setLocalFps] = useState(config.target_fps);

  const handleFpsChange = async (fps) => {
    setLocalFps(fps);
    try {
      const data = await api.setTargetFps(fps);
      setConfig({ target_fps: data.target_fps });
    } catch {}
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-white">Target FPS</h3>
        <span className="text-lg font-mono font-bold text-neon-cyan">{localFps}</span>
      </div>

      <input
        type="range"
        min="1"
        max="60"
        value={localFps}
        onChange={e => handleFpsChange(parseInt(e.target.value))}
        disabled={config.auto_fps}
        className="w-full h-1.5 bg-border rounded-lg appearance-none cursor-pointer accent-neon-cyan disabled:opacity-30"
      />

      <div className="flex justify-between text-xs font-mono text-gray-600">
        <span>1 FPS</span>
        <span>30 FPS</span>
        <span>60 FPS</span>
      </div>

      {config.auto_fps && (
        <p className="text-xs text-neon-amber font-mono">
          Auto FPS is enabled. Disable it in the header to set manually.
        </p>
      )}
    </div>
  );
}
